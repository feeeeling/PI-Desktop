/**
 * Vendor-account (OAuth) login for model providers.
 *
 * pi-ai owns all supported login flows and the locked token refresh; persistence
 * and the user-facing half of the conversation are the app's job (see
 * `auth/types.d.ts`: "Login/account-removal orchestration is app-owned"). This module is
 * that half:
 *
 *  - a CredentialStore backed by host-core's encrypted secret store, keyed
 *    `secret:provider:<providerRowId>:oauth` so an API key and a vendor account
 *    can coexist on one provider row;
 *  - a bridge from pi-ai's prompt/notify interaction to renderer events;
 *  - short-lived request auth (`ModelAuth`) for the agent sidecar.
 *
 * Refresh tokens never leave the main process: callers get either a boolean, a
 * non-secret account label, or an already-resolved `ModelAuth`.
 */

import { randomUUID } from "node:crypto";

import { InMemoryModelsStore } from "@earendil-works/pi-ai";
import type {
  Api,
  AuthEvent,
  AuthInteraction,
  AuthPrompt,
  Credential,
  CredentialStore,
  Model,
  ModelAuth,
  MutableModels,
  Provider,
} from "@earendil-works/pi-ai";
import { registerBunOAuthFlows } from "@earendil-works/pi-ai/bun-oauth";
import { builtinModels } from "@earendil-works/pi-ai/providers/all";
import {
  capabilitiesFromModelConfig,
  genericModelConfig,
  installProviderHeadersFetch,
  runWithProviderHeaders,
  type ModelConfig,
  type VendorModelBinding,
} from "@pi-desktop/agent-runtime";
import { modelListRequest } from "./model-discovery.ts";
import {
  OAUTH_AUTH_KIND,
  type OAuthLoginEvent,
  type OAuthPromptRequest,
  type OAuthRespondInput,
  type OAuthStartResult,
  type OAuthVendor,
  type ModelBinding,
  type ThinkingLevel,
} from "@pi-desktop/shared";

export { OAUTH_AUTH_KIND };

/** Mirrors `secret_ref_for_provider_oauth` in crates/host-core/src/secrets.rs. */
export function secretRefForProviderOauth(providerId: string): string {
  return `secret:provider:${providerId}:oauth`;
}

const API_STYLE_BY_WIRE_API: Record<string, string> = {
  "anthropic-messages": "anthropic_messages",
  "openai-completions": "chat_completions",
  "openai-responses": "responses",
  "openai-codex-responses": "openai_codex_responses",
  "google-generative-ai": "google_generative_ai",
  "pi-messages": "pi_messages",
};

const PROTOCOL_BY_API_STYLE: Record<string, string> = {
  anthropic_messages: "anthropic",
  chat_completions: "openai_compatible",
  responses: "openai",
  openai_codex_responses: "openai",
  google_generative_ai: "google",
  pi_messages: "custom_http",
};

/**
 * A provider row stores one apiStyle, but a vendor can span wire APIs (GitHub
 * Copilot serves Anthropic, Chat Completions and Responses models), so the
 * style follows the selected model rather than the vendor.
 */
export function apiStyleForWireApi(api: string): string {
  return API_STYLE_BY_WIRE_API[api] ?? "chat_completions";
}

export function protocolForApiStyle(apiStyle: string): string {
  return PROTOCOL_BY_API_STYLE[apiStyle] ?? "openai_compatible";
}

const LIVE_MODELS_TTL_MS = 30_000;
const THINKING_LEVEL_ORDER = [
  "off",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
] as const satisfies readonly ThinkingLevel[];

/**
 * Vendor `/models` payloads also publish image, video, speech and embedding
 * models. Those are not conversation models; the chat picker and session
 * launch both reject them.
 */
const NON_CONVERSATION_MODEL =
  /(?:^|[-_/])(?:imagine|image|video|tts|stt|embed(?:ding)?|whisper|aurora|flux)(?:$|[-_/])/i;

export function isConversationModel(modelId: string): boolean {
  const id = modelId.trim();
  return id.length > 0 && !NON_CONVERSATION_MODEL.test(id);
}

/** @deprecated Use {@link isConversationModel}. Kept so callers of the xAI fix still compile. */
export function isXaiConversationModel(modelId: string): boolean {
  return isConversationModel(modelId);
}

/** Leading letter run: `claude-opus-4.8` and `gpt-5.5` belong to different wire APIs. */
export function modelFamily(modelId: string): string {
  return /^[a-z]+/i.exec(modelId.trim())?.[0]?.toLowerCase() ?? modelId.trim().toLowerCase();
}

const COPILOT_LIST_HEADERS: Record<string, string> = {
  Accept: "application/json",
  "User-Agent": "GitHubCopilotChat/0.35.0",
  "Editor-Version": "vscode/1.107.0",
  "Editor-Plugin-Version": "copilot-chat/0.35.0",
  "Copilot-Integration-Id": "vscode-chat",
};

/**
 * Where to ask a signed-in account which models it serves.
 * One wire API uses that API's list endpoint. Mixed catalogs (Copilot) use
 * the account's OpenAI-compatible `/models`.
 */
export function accountModelListRequest(input: {
  vendorId: string;
  api: string | undefined;
  baseUrl: string;
  apiKey: string;
}): { url: string; headers: Record<string, string> } {
  const base = input.baseUrl.trim().replace(/\/+$/, "");
  if (input.vendorId === "github-copilot") {
    return {
      url: `${base}/models`,
      headers: { ...COPILOT_LIST_HEADERS, Authorization: `Bearer ${input.apiKey}` },
    };
  }
  if (input.api === "anthropic-messages" && input.apiKey.includes("sk-ant-oat")) {
    const root = base.endsWith("/v1") ? base : `${base}/v1`;
    return {
      url: `${root}/models?limit=1000`,
      headers: {
        Accept: "application/json",
        Authorization: `Bearer ${input.apiKey}`,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
      },
    };
  }
  return modelListRequest({
    baseUrl: base,
    apiKey: input.apiKey,
    apiStyle: input.api ? apiStyleForWireApi(input.api) : "chat_completions",
  });
}

type PinnedWire = { id: string; api: string; baseUrl: string };

/** Map one live id onto a wire API. Unknown families on a mixed catalog are dropped. */
export function wireForAccountModel(
  modelId: string,
  pinned: readonly PinnedWire[],
  authBaseUrl?: string,
): { api: string; baseUrl: string } | undefined {
  const known = pinned.find((model) => model.id === modelId);
  if (known) {
    return { api: known.api, baseUrl: authBaseUrl || known.baseUrl };
  }
  const apis = new Set(pinned.map((model) => model.api));
  const bases = new Set(pinned.map((model) => model.baseUrl));
  if (apis.size === 1 && bases.size === 1) {
    return { api: [...apis][0]!, baseUrl: authBaseUrl || [...bases][0]! };
  }
  const family = modelFamily(modelId);
  const matches = pinned.filter((model) => modelFamily(model.id) === family);
  const matchApis = new Set(matches.map((model) => model.api));
  const matchBases = new Set(matches.map((model) => model.baseUrl));
  if (matchApis.size !== 1 || matchBases.size !== 1) return undefined;
  return { api: [...matchApis][0]!, baseUrl: authBaseUrl || [...matchBases][0]! };
}

function asModelRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

/** Conversation ids from an OpenAI- or Anthropic-shaped model list. */
export function accountModelIds(body: unknown): string[] {
  const record = asModelRecord(body);
  const data = Array.isArray(record?.data)
    ? record.data
    : Array.isArray(body)
      ? body
      : [];
  const pickerScoped = data.some((entry) => {
    const item = asModelRecord(entry);
    return typeof item?.model_picker_enabled === "boolean";
  });
  const ids: string[] = [];
  for (const entry of data) {
    const item = asModelRecord(entry);
    const id = item?.id;
    if (!item || typeof id !== "string" || !isConversationModel(id)) continue;
    if (pickerScoped && item.model_picker_enabled !== true) continue;
    const capabilities = asModelRecord(item.capabilities);
    const supports = asModelRecord(capabilities?.supports);
    if (supports?.tool_calls === false) continue;
    const policy = asModelRecord(item.policy);
    if (policy?.state === "disabled") continue;
    ids.push(id);
  }
  return [...new Set(ids)].sort((left, right) => left.localeCompare(right));
}

export function accountModelOptions(
  body: unknown,
  pinned: readonly Model<Api>[],
  authBaseUrl?: string,
): OAuthModelOption[] {
  const wires: PinnedWire[] = pinned.map((model) => ({
    id: model.id,
    api: model.api,
    baseUrl: model.baseUrl,
  }));
  const options: OAuthModelOption[] = [];
  for (const modelId of accountModelIds(body)) {
    const wire = wireForAccountModel(modelId, wires, authBaseUrl);
    if (!wire) continue;
    options.push({
      modelId,
      apiStyle: apiStyleForWireApi(wire.api),
      baseUrl: wire.baseUrl,
    });
  }
  return options;
}

async function fetchAccountModelList(request: {
  url: string;
  headers: Record<string, string>;
}): Promise<unknown> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 10_000);
  try {
    const response = await fetch(request.url, { headers: request.headers, signal: controller.signal });
    if (!response.ok) {
      throw new Error(`model list request failed (${response.status})`);
    }
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
}

function thinkingLevelsFromPiModel(model: Model<Api>): ThinkingLevel[] {
  const map = model.thinkingLevelMap as Partial<Record<string, string | null>> | undefined;
  if (!map) return model.reasoning ? ["low", "medium", "high"] : ["off"];
  const levels = THINKING_LEVEL_ORDER.filter((level) => typeof map[level] === "string");
  return levels.length > 0
    ? [...levels]
    : model.reasoning
      ? ["low", "medium", "high"]
      : ["off"];
}

export type HostCall = <T = unknown>(
  method: string,
  params?: unknown,
) => Promise<T>;

/** The slice of a provider row this module reads; the rest stays in index.ts. */
export type OAuthProviderRow = {
  id: string;
  vendorKey?: string;
  authKind?: string;
  hasOauth?: boolean;
  oauthAccountLabel?: string;
  headers?: Record<string, string>;
  baseUrl?: string;
  defaultModelId?: string;
};

/** A model ID offered by a signed-in account and its required wire identity. */
export type OAuthModelOption = {
  modelId: string;
  apiStyle: string;
  baseUrl: string;
};

export type VendorOAuthDeps = {
  /** host-core RPC. Only the generic `secrets.*` and `providers.*` methods. */
  call: HostCall;
  /** Push login progress to the renderer. Never carries token material. */
  emit: (event: OAuthLoginEvent) => void;
  /** Open the vendor's consent page; rejects when no browser could be launched. */
  openExternal: (url: string) => Promise<void>;
  log?: (
    level: "info" | "warn" | "error",
    message: string,
    data?: Record<string, unknown>,
  ) => void;
  /** Test seam: build the pi-ai collection without touching the real flows. */
  createModels?: (credentials: CredentialStore) => MutableModels;
  /** Model configuration is supplied by the main-process models.dev catalog. */
  modelConfigFor?: (input: {
    vendorKey: string;
    option: OAuthModelOption;
  }) => Promise<ModelConfig | undefined>;
  /**
   * Test seam for the account model-list request. Production fetches the URL.
   * Throw to force the pinned-catalog fallback.
   */
  discoverAccountModels?: (input: {
    vendorId: string;
    url: string;
    headers: Record<string, string>;
  }) => Promise<unknown>;
  newId?: () => string;
};

/**
 * A login event minus the identifiers `push()` stamps on. Distributed over the
 * union so each variant keeps its own fields.
 */
type OAuthEventBody = OAuthLoginEvent extends infer Variant
  ? Variant extends unknown
    ? Omit<Variant, "loginId" | "vendorId">
    : never
  : never;

type PendingPrompt = {
  resolve: (value: string) => void;
  reject: (error: Error) => void;
};

type AccountModels = {
  providerId: string;
  vendorId: string;
  models: MutableModels;
};

type LoginSession = {
  loginId: string;
  vendorId: string;
  providerId: string;
  account: AccountModels;
  /** Whether this login created the row, and so owns cleaning it up on failure. */
  createdRow: boolean;
  controller: AbortController;
  prompts: Map<string, PendingPrompt>;
  /** Serializes renderer events so `authUrl` cannot overtake an earlier notice. */
  tail: Promise<void>;
  /** Settles once the attempt has torn down; set as soon as it is running. */
  finished?: Promise<void>;
};

function promptRequest(
  promptId: string,
  prompt: AuthPrompt,
): OAuthPromptRequest {
  return {
    promptId,
    type: prompt.type,
    message: prompt.message,
    placeholder: "placeholder" in prompt ? prompt.placeholder : undefined,
    options:
      prompt.type === "select"
        ? prompt.options.map((option) => ({
            id: option.id,
            label: option.label,
            description: option.description,
          }))
        : undefined,
  };
}

export class VendorOAuth {
  private readonly deps: VendorOAuthDeps;
  private readonly logins = new Map<string, LoginSession>();
  /** One pi-ai collection and credential store per local OAuth account row. */
  private readonly accountModels = new Map<string, AccountModels>();
  /** Successful account `/models` responses, so one login does not refetch per model. */
  private readonly liveModels = new Map<string, { at: number; models: OAuthModelOption[] }>();
  /** Per-account write chain: `modify` must be a serialized read-modify-write. */
  private readonly chains = new Map<string, Promise<unknown>>();
  private catalogPromise?: Promise<MutableModels>;
  private oauthFlowsRegistered = false;

  constructor(deps: VendorOAuthDeps) {
    this.deps = deps;
    installProviderHeadersFetch();
  }

  /** Every vendor pi-ai can sign in to, with every local account row. */
  async listVendors(): Promise<OAuthVendor[]> {
    const models = await this.ensureCatalogModels();
    const rows = await this.rows();
    return models
      .getProviders()
      .filter((provider) => provider.auth.oauth)
      .map((provider) => {
        const oauth = provider.auth.oauth!;
        const accounts = rows
          .filter(
            (candidate) =>
              candidate.authKind === OAUTH_AUTH_KIND &&
              candidate.vendorKey === provider.id,
          )
          .map((row) => ({
            providerId: row.id,
            accountLabel: row.oauthAccountLabel || undefined,
            connected: row.hasOauth === true,
          }));
        return {
          vendorId: provider.id,
          name: oauth.name || provider.name,
          loginLabel: oauth.loginLabel,
          isSubscription: oauth.isSubscription === true,
          accounts,
        };
      });
  }

  /**
   * Begin a login. Every attempt gets a fresh provider row and credential
   * store, so signing into the same vendor twice creates two independent
   * accounts instead of silently replacing the first one.
   */
  async start(vendorId: string): Promise<OAuthStartResult> {
    const models = await this.ensureCatalogModels();
    const provider = models.getProvider(vendorId);
    if (!provider?.auth.oauth) {
      throw new Error(`unknown vendor account: ${vendorId}`);
    }
    // A second attempt replaces the one in flight rather than racing it, and
    // waits for it to let go: both hold the same local callback port, so
    // starting before the old one unwinds is how a login fails on arrival.
    const superseded = [...this.logins.values()].filter(
      (running) => running.vendorId === vendorId,
    );
    for (const running of superseded) this.cancel(running.loginId);
    for (const running of superseded) {
      await running.finished?.catch(() => undefined);
    }

    const { provider: row } = await this.deps.call<{
      provider: OAuthProviderRow;
    }>("providers.create", {
      name: provider.auth.oauth?.name || provider.name,
      vendorKey: vendorId,
      type: "native",
      authKind: OAUTH_AUTH_KIND,
      baseUrl: provider.baseUrl,
    });
    const account = this.createAccount(vendorId, row.id);
    const session: LoginSession = {
      loginId: this.nextId(),
      vendorId,
      providerId: row.id,
      account,
      createdRow: true,
      controller: new AbortController(),
      prompts: new Map(),
      tail: Promise.resolve(),
    };
    this.logins.set(session.loginId, session);
    session.finished = this.run(session, provider);
    return { loginId: session.loginId };
  }

  /** Answer a prompt. An absent value cancels the prompt and the login. */
  respond(input: OAuthRespondInput): boolean {
    const session = this.logins.get(input.loginId);
    const pending = session?.prompts.get(input.promptId);
    if (!session || !pending) return false;
    if (input.value === undefined) {
      pending.reject(new Error("login cancelled"));
      session.controller.abort();
    } else {
      pending.resolve(input.value);
    }
    return true;
  }

  /** Abort a login: stops the local callback server or device-code polling. */
  cancel(loginId: string): boolean {
    const session = this.logins.get(loginId);
    if (!session) return false;
    session.controller.abort();
    return true;
  }

  /**
   * Delete one account's provider row and its provider-scoped OAuth secret.
   * Host-core owns the atomic cleanup of the row and both secret references.
   */
  async deleteAccount(providerId: string): Promise<void> {
    const row = (await this.rows()).find((candidate) => candidate.id === providerId);
    if (!row || row.authKind !== OAUTH_AUTH_KIND) {
      throw new Error(`unknown vendor account provider: ${providerId}`);
    }
    const running = [...this.logins.values()].find(
      (session) => session.providerId === providerId,
    );
    if (running) {
      this.cancel(running.loginId);
      await running.finished?.catch(() => undefined);
    }
    this.accountModels.delete(providerId);
    this.liveModels.delete(providerId);
    await this.deps.call("providers.delete", { id: providerId });
  }

  /**
   * Resolve request auth for one model request. pi-ai refreshes the token under
   * the store lock when it has expired; the caller only ever sees the resulting
   * short-lived access token, headers and per-credential baseUrl.
   */
  async resolveAuth(providerId: string): Promise<ModelAuth> {
    return this.withRowHeaders(providerId, async () => {
      const account = await this.accountForProvider(providerId);
      if (!account) throw new Error(`vendor account not signed in: ${providerId}`);
      const resolved = await account.models.getAuth(account.vendorId);
      if (!resolved) throw new Error(`vendor account not signed in: ${providerId}`);
      return resolved.auth;
    });
  }

  /**
   * Models the signed-in account may actually use.
   *
   * A pinned pi-ai catalog goes stale the day the vendor ships a model, which
   * is why selecting it used to require a client update. When the account
   * token can read `/models`, that chat-model list is the authority. Vendors
   * that already publish a live catalog (Radius) are left alone. A failed
   * probe keeps `getAvailable`, including Copilot's subscription filter.
   */
  async listModels(providerId: string): Promise<OAuthModelOption[]> {
    return this.withRowHeaders(providerId, async () => {
      const account = await this.accountForProvider(providerId);
      if (!account) throw new Error(`unknown vendor account provider: ${providerId}`);
      // Dynamic catalogs (radius) are empty until refreshed; static and
      // unconfigured providers are skipped inside pi-ai.
      await account.models.refresh({ providers: [account.vendorId] });
      const live = await this.liveAccountModels(account);
      if (live) return live;
      const available = await account.models.getAvailable(account.vendorId);
      return available.map((model) => this.optionFor(model));
    });
  }

  private optionFor(model: Model<Api>): OAuthModelOption {
    return {
      modelId: model.id,
      apiStyle: apiStyleForWireApi(model.api),
      baseUrl: model.baseUrl,
    };
  }

  /**
   * Everything a provider binding needs for one model of a signed-in account.
   *
   * A vendor row cannot take this from the builtin catalog: one account spans
   * wire APIs (GitHub Copilot serves Anthropic, Chat Completions and Responses
   * models, so the selected model — not the row — decides the style), and a
   * gateway's catalog (radius) is not in the builtin one at all. The
   * authenticated collection knows both.
   */
  async bindingFor(
    providerId: string,
    modelId: string,
  ): Promise<VendorModelBinding | undefined> {
    return this.withRowHeaders(providerId, () =>
      this.bindingForUnscoped(providerId, modelId),
    );
  }

  private async bindingForUnscoped(
    providerId: string,
    modelId: string,
  ): Promise<VendorModelBinding | undefined> {
    const account = await this.accountForProvider(providerId);
    if (!account) return undefined;
    let model = account.models.getModel(account.vendorId, modelId);
    if (!model) {
      // Dynamic catalogs are empty until the first refresh.
      await account.models.refresh({ providers: [account.vendorId] });
      model = account.models.getModel(account.vendorId, modelId);
    }
    const live = await this.liveAccountModels(account);
    if (live) {
      const option = live.find((item) => item.modelId === modelId);
      // A successful account list replaces the pinned catalog. An id it did
      // not return is not offered, even when pi-ai still ships that id.
      if (!option) return undefined;
      return this.bindingFromOption(account, option);
    }
    if (!model) return undefined;
    return this.bindingFromOption(account, this.optionFor(model));
  }

  /**
   * Chat models this signed-in account can call right now.
   * `undefined` means the live list could not be read or the vendor already
   * refreshes its own catalog; callers keep `getAvailable`.
   */
  private async liveAccountModels(
    account: AccountModels,
  ): Promise<OAuthModelOption[] | undefined> {
    const provider = account.models.getProvider(account.vendorId);
    if (typeof provider?.refreshModels === "function") return undefined;
    const cached = this.liveModels.get(account.providerId);
    if (cached && Date.now() - cached.at < LIVE_MODELS_TTL_MS) return cached.models;
    const pinned = await this.pinnedCatalog(account);
    if (pinned.length === 0) return undefined;
    let apiKey: string | undefined;
    let authBaseUrl: string | undefined;
    try {
      const resolved = await account.models.getAuth(account.vendorId);
      apiKey = resolved?.auth.apiKey;
      if (resolved?.auth.baseUrl) authBaseUrl = resolved.auth.baseUrl.replace(/\/+$/, "");
    } catch (error) {
      this.log("warn", "account auth unavailable for model list", {
        vendorId: account.vendorId,
        message: error instanceof Error ? error.message : String(error),
      });
      return undefined;
    }
    if (!apiKey) return undefined;
    const listBase = authBaseUrl || pinned[0]?.baseUrl;
    if (!listBase) return undefined;
    const apis = new Set(pinned.map((model) => model.api));
    const request = accountModelListRequest({
      vendorId: account.vendorId,
      api: apis.size === 1 ? pinned[0]!.api : undefined,
      baseUrl: listBase,
      apiKey,
    });
    try {
      const body = this.deps.discoverAccountModels
        ? await this.deps.discoverAccountModels({
            vendorId: account.vendorId,
            url: request.url,
            headers: request.headers,
          })
        : await fetchAccountModelList(request);
      const models = accountModelOptions(body, pinned, authBaseUrl);
      if (models.length === 0) return undefined;
      this.liveModels.set(account.providerId, { at: Date.now(), models });
      return models;
    } catch (error) {
      this.log("warn", "account model list failed", {
        vendorId: account.vendorId,
        message: error instanceof Error ? error.message : String(error),
      });
      return undefined;
    }
  }

  private async pinnedCatalog(account: AccountModels): Promise<Model<Api>[]> {
    const provider = account.models.getProvider(account.vendorId);
    if (typeof provider?.getModels === "function") return [...provider.getModels()];
    return [...(await account.models.getAvailable(account.vendorId))];
  }

  private async bindingFromOption(
    account: AccountModels,
    option: OAuthModelOption,
  ): Promise<VendorModelBinding> {
    const published = await this.deps.modelConfigFor?.({
      vendorKey: account.vendorId,
      option,
    }).catch(() => undefined);
    const modelConfig = await this.withPinnedSiblingFallback(account, option, published);
    const capabilities = capabilitiesFromModelConfig(modelConfig);
    return {
      apiStyle: option.apiStyle,
      baseUrl: option.baseUrl,
      modelConfig,
      ...capabilities,
    };
  }

  /**
   * models.dev is the metadata source when it already knows the id. A model
   * that exists only on the live account list otherwise inherits limits and
   * thinking levels from a pinned sibling of the same family, instead of the
   * 128k generic shape.
   */
  private async withPinnedSiblingFallback(
    account: AccountModels,
    option: OAuthModelOption,
    published: ModelConfig | undefined,
  ): Promise<ModelConfig> {
    const config = published ?? genericModelConfig(option.modelId, option.baseUrl);
    if (config.source !== "generic") return config;
    const pinned = await this.pinnedCatalog(account);
    if (pinned.some((model) => model.id === option.modelId)) return config;
    const family = modelFamily(option.modelId);
    const sibling = pinned.find((model) => modelFamily(model.id) === family)
      ?? pinned.find((model) => apiStyleForWireApi(model.api) === option.apiStyle);
    if (!sibling) return config;
    const input = (sibling.input ?? []).filter(
      (modality): modality is "text" | "image" => modality === "text" || modality === "image",
    );
    return {
      ...config,
      reasoning: sibling.reasoning,
      input: input.length > 0 ? input : config.input,
      contextWindow: sibling.contextWindow,
      maxTokens: sibling.maxTokens,
      limit: {
        context: sibling.contextWindow,
        input: sibling.contextWindow,
        output: sibling.maxTokens,
      },
      supportedThinkingLevels: thinkingLevelsFromPiModel(sibling),
    };
  }

  private async run(session: LoginSession, provider: Provider): Promise<void> {
    try {
      await this.withRowHeaders(session.providerId, () =>
        session.account.models.login(
          session.vendorId,
          "oauth",
          this.interactionFor(session),
        ),
      );
      const accountLabel = provider.auth.oauth?.name || provider.name;
      await this.completeRow(session, provider, accountLabel);
      this.push(session, {
        kind: "done",
        providerId: session.providerId,
        accountLabel,
      });
    } catch (error) {
      await this.discardRow(session);
      if (session.controller.signal.aborted) {
        this.push(session, { kind: "cancelled" });
      } else {
        const message = error instanceof Error ? error.message : String(error);
        this.log("warn", "vendor account login failed", {
          vendorId: session.vendorId,
          message,
        });
        this.push(session, { kind: "error", message });
      }
    } finally {
      for (const pending of session.prompts.values()) {
        pending.reject(new Error("login finished"));
      }
      await session.tail;
      this.logins.delete(session.loginId);
    }
  }

  /** Point the row at a usable model now that the catalog can be read. */
  private async completeRow(
    session: LoginSession,
    provider: Provider,
    accountLabel: string,
  ): Promise<void> {
    let options: OAuthModelOption[] = [];
    try {
      options = await this.listModels(session.providerId);
    } catch (error) {
      // A catalog that will not load is not worth failing a good login over;
      // the row stays selectable and the model picker retries later.
      this.log("warn", "vendor model catalog unavailable after login", {
        vendorId: session.vendorId,
        message: error instanceof Error ? error.message : String(error),
      });
    }
    const chosen = options[0];
    const apiStyle = chosen?.apiStyle;
    const modelBindings: ModelBinding[] = [];
    for (const option of options) {
      const binding = await this.bindingFor(session.providerId, option.modelId).catch(
        () => undefined,
      );
      const levels = binding?.supportedThinkingLevels ?? ["off"];
      modelBindings.push({
        id: option.modelId,
        contextWindow: binding?.modelConfig.contextWindow ?? 128_000,
        maxTokens: binding?.modelConfig.maxTokens ?? 8_192,
        thinkingLevels: [...levels],
        defaultThinkingLevel: levels.includes("medium") ? "medium" : levels[0] ?? null,
      });
    }
    await this.deps.call("providers.update", {
      id: session.providerId,
      name: provider.auth.oauth?.name || provider.name,
      authKind: OAUTH_AUTH_KIND,
      oauthAccountLabel: accountLabel,
      baseUrl: chosen?.baseUrl ?? provider.baseUrl,
      ...(modelBindings.length > 0 ? { models: modelBindings } : {}),
      ...(apiStyle
        ? {
            apiStyle,
            protocol: protocolForApiStyle(apiStyle),
            defaultModelId: chosen?.modelId,
          }
        : {}),
    });
  }

  private async discardRow(session: LoginSession): Promise<void> {
    if (!session.createdRow) return;
    this.accountModels.delete(session.providerId);
    try {
      await this.deps.call("providers.delete", { id: session.providerId });
    } catch (error) {
      this.log("warn", "could not remove the half-created provider row", {
        vendorId: session.vendorId,
        message: error instanceof Error ? error.message : String(error),
      });
    }
  }

  private interactionFor(session: LoginSession): AuthInteraction {
    return {
      signal: session.controller.signal,
      prompt: (prompt) => this.ask(session, prompt),
      notify: (event) => this.notify(session, event),
    };
  }

  private ask(session: LoginSession, prompt: AuthPrompt): Promise<string> {
    const promptId = this.nextId();
    return new Promise<string>((resolve, reject) => {
      const cleanups: Array<() => void> = [];
      const settle = () => {
        session.prompts.delete(promptId);
        for (const cleanup of cleanups) cleanup();
      };
      const entry: PendingPrompt = {
        resolve: (value) => {
          settle();
          resolve(value);
        },
        reject: (error) => {
          settle();
          reject(error);
        },
      };
      session.prompts.set(promptId, entry);

      // The flow cancels a prompt when it answers the step itself — a callback
      // that beats the paste box — so the renderer has to close that input.
      const abort = () => {
        this.push(session, { kind: "promptCancelled", promptId });
        entry.reject(new Error("prompt cancelled"));
      };
      for (const signal of [prompt.signal, session.controller.signal]) {
        if (!signal) continue;
        if (signal.aborted) {
          abort();
          return;
        }
        signal.addEventListener("abort", abort, { once: true });
        cleanups.push(() => signal.removeEventListener("abort", abort));
      }

      this.push(session, {
        kind: "prompt",
        request: promptRequest(promptId, prompt),
      });
    });
  }

  private notify(session: LoginSession, event: AuthEvent): void {
    switch (event.type) {
      case "info":
        this.push(session, {
          kind: "info",
          message: event.message,
          links: event.links?.map((link) => ({
            url: link.url,
            label: link.label,
          })),
        });
        return;
      case "auth_url":
        this.pushAuthUrl(session, event.url, event.instructions);
        return;
      case "device_code":
        this.push(session, {
          kind: "deviceCode",
          userCode: event.userCode,
          verificationUri: event.verificationUri,
          intervalSeconds: event.intervalSeconds,
          expiresInSeconds: event.expiresInSeconds,
        });
        return;
      case "progress":
        this.push(session, { kind: "progress", message: event.message });
    }
  }

  private pushAuthUrl(
    session: LoginSession,
    url: string,
    instructions?: string,
  ): void {
    session.tail = session.tail.then(async () => {
      // Report whether the browser actually opened: when it did not, the
      // renderer has to offer the link for copying instead.
      const opened = await this.deps.openExternal(url).then(
        () => true,
        () => false,
      );
      this.deps.emit({
        loginId: session.loginId,
        vendorId: session.vendorId,
        kind: "authUrl",
        url,
        instructions,
        opened,
      });
    });
  }

  private push(session: LoginSession, event: OAuthEventBody): void {
    session.tail = session.tail.then(() => {
      this.deps.emit({
        ...event,
        loginId: session.loginId,
        vendorId: session.vendorId,
      } as OAuthLoginEvent);
    });
  }

  private async ensureCatalogModels(): Promise<MutableModels> {
    this.catalogPromise ??= Promise.resolve(
      this.createModels(this.emptyCredentials),
    );
    return this.catalogPromise;
  }

  private createModels(credentials: CredentialStore): MutableModels {
    if (this.deps.createModels) return this.deps.createModels(credentials);
    // pi-ai loads each flow through a variable import specifier so bundlers
    // cannot follow it into Node-only code; registering the static set keeps
    // login working in the packaged app. Named for the Bun binary, but the
    // flows themselves are plain Node.
    if (!this.oauthFlowsRegistered) {
      registerBunOAuthFlows();
      this.oauthFlowsRegistered = true;
    }
    return builtinModels({
      credentials,
      modelsStore: new InMemoryModelsStore(),
    });
  }

  private createAccount(vendorId: string, providerId: string): AccountModels {
    const existing = this.accountModels.get(providerId);
    if (existing) return existing;
    const account = {
      providerId,
      vendorId,
      models: this.createModels(this.credentialsFor(providerId, vendorId)),
    };
    this.accountModels.set(providerId, account);
    return account;
  }

  private async withRowHeaders<T>(
    providerId: string,
    fn: () => Promise<T>,
  ): Promise<T> {
    const row = (await this.rows()).find((candidate) => candidate.id === providerId);
    return await runWithProviderHeaders(row?.headers, fn);
  }

  private async accountForProvider(
    providerId: string,
  ): Promise<AccountModels | undefined> {
    const existing = this.accountModels.get(providerId);
    if (existing) return existing;
    const row = (await this.rows()).find(
      (candidate) =>
        candidate.id === providerId &&
        candidate.authKind === OAUTH_AUTH_KIND &&
        typeof candidate.vendorKey === "string" &&
        candidate.vendorKey.length > 0,
    );
    return row?.vendorKey
      ? this.createAccount(row.vendorKey, row.id)
      : undefined;
  }

  /**
   * Create a store scoped to one provider row. pi-ai still addresses the
   * credential by its builtin vendor id, while the app maps that id to the
   * row-specific encrypted secret ref.
   */
  private credentialsFor(
    providerId: string,
    vendorId: string,
  ): CredentialStore {
    return {
      read: (requestedVendorId) => {
        if (requestedVendorId !== vendorId) return Promise.resolve(undefined);
        return this.readCredential(providerId);
      },
      list: async () => {
        const row = (await this.rows()).find(
          (candidate) => candidate.id === providerId,
        );
        return row?.authKind === OAUTH_AUTH_KIND && row.hasOauth
          ? [{ providerId: vendorId, type: "oauth" }]
          : [];
      },
      modify: (requestedVendorId, fn) => {
        if (requestedVendorId !== vendorId) {
          throw new Error(`provider is not part of account ${providerId}`);
        }
        return this.serialize(providerId, async () => {
          const current = await this.readCredential(providerId);
          const next = await fn(current);
          if (next === undefined) return current;
          await this.deps.call("secrets.set", {
            secretRef: secretRefForProviderOauth(providerId),
            value: JSON.stringify(next),
          });
          return next;
        });
      },
      delete: (requestedVendorId) => {
        if (requestedVendorId !== vendorId) {
          throw new Error(`provider is not part of account ${providerId}`);
        }
        return this.serialize(providerId, async () => {
          await this.deps.call("secrets.delete", {
            secretRef: secretRefForProviderOauth(providerId),
          });
        });
      },
    };
  }

  private readonly emptyCredentials: CredentialStore = {
    read: async () => undefined,
    list: async () => [],
    modify: async (_providerId, fn) => fn(undefined),
    delete: async () => undefined,
  };

  private async readCredential(
    providerId: string,
  ): Promise<Credential | undefined> {
    const { value } = await this.deps.call<{ value?: string | null }>(
      "secrets.getForRuntime",
      { secretRef: secretRefForProviderOauth(providerId) },
    );
    if (!value) return undefined;
    try {
      const parsed = JSON.parse(value) as Credential;
      return parsed?.type ? parsed : undefined;
    } catch {
      this.log("warn", "stored vendor credential is not readable", { providerId });
      return undefined;
    }
  }

  private serialize<T>(key: string, fn: () => Promise<T>): Promise<T> {
    const previous = this.chains.get(key) ?? Promise.resolve();
    const next = previous.then(fn, fn);
    this.chains.set(
      key,
      next.then(
        () => undefined,
        () => undefined,
      ),
    );
    return next;
  }

  private async rows(): Promise<OAuthProviderRow[]> {
    const result = await this.deps.call<{ providers?: OAuthProviderRow[] }>(
      "providers.list",
      { includeDisabled: true },
    );
    return result.providers ?? [];
  }

  private nextId(): string {
    return this.deps.newId?.() ?? randomUUID();
  }

  private log(
    level: "info" | "warn" | "error",
    message: string,
    data?: Record<string, unknown>,
  ): void {
    this.deps.log?.(level, message, data);
  }
}
