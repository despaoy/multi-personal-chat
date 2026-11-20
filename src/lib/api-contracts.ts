/**
 * Shared HTTP contract types.
 *
 * Keep transport-independent request and response shapes here so API methods,
 * hooks, and pages consume one source of truth. Runtime validation remains the
 * responsibility of the backend.
 */

export type JsonRecord = Record<string, unknown>;

export interface CharacterSettings {
  name: string;
  persona: string;
  greeting: string;
  scenario: string;
  relationships: string;
  examples: Array<{ role: 'user' | 'assistant'; content: string }>;
  reference_text: string;
}

export interface CharacterLibraryRecord {
  id: string;
  revision: number;
  settings_revision: number;
  settings: CharacterSettings;
  share_token: string | null;
  shared_revision: number | null;
  publication_id: string | null;
  source_publication_id: string | null;
  source_character_id: string | null;
  source_revision: number | null;
  created_at: string;
  updated_at: string;
}

export interface CharacterPackage {
  format: 'multipersonal-character-v1';
  settings: CharacterSettings;
}

export interface CharacterSourceUpdate {
  available: boolean;
  revision: number;
  source_revision: number | null;
  changes: Record<string, { current: unknown; incoming: unknown }>;
  publication_id: string | null;
  local_corpus_id: string | null;
  sources: CharacterOriginalSource[];
  current_sources: CharacterOriginalSource[];
  incoming_sources: CharacterOriginalSource[];
  baseline_publication_id: string | null;
  source_merge: CharacterSourceMerge;
}

export type SourceChoices = Record<string, 'local' | 'author'>;
export interface CharacterSourceMerge {
  status: 'ready' | 'conflict' | 'blocked';
  reason: string | null;
  actions: Array<{ id: string; action: string }>;
  conflicts: Array<{ id: string; kind: string }>;
  merged_sources: CharacterOriginalSource[];
  resolved_conflicts?: Array<{ id: string; kind: string; choice: 'local' | 'author' }>;
}
export interface SourceUpdateRequest {
  revision: number;
  source_revision: number;
  publication_id: string;
  local_corpus_id: string | null;
  choices: SourceChoices;
}
export interface FinalSourceUpdatePreview extends SourceUpdateRequest {
  preview_id: string;
  baseline_publication_id: string;
  settings: CharacterSettings;
  sources: CharacterOriginalSource[];
  source_merge: CharacterSourceMerge;
}

export interface CharacterPublicationPreview {
  character_id: string;
  revision: number;
  corpus_id: string | null;
  settings: CharacterSettings;
  sources: CharacterOriginalSource[];
  preview_id: string;
  previous_publication_id: string | null;
}

export interface SharedCharacterPublication {
  id: string;
  settings_revision: number;
  publication_id: string;
  settings: CharacterSettings;
  sources: CharacterOriginalSource[];
}

export interface CharacterChatSummary {
  id: string;
  character_id: string;
  character_revision: number;
  title: string;
  revision: number;
  mode: 'initial' | 'continuation' | 'branch';
  source_session_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface CharacterChatEvidence {
  sources?: Array<{ id: string; title: string; excerpt: string; truncated: boolean; line_start?: number | null; line_end?: number | null }>;
  retrieval_status?: string;
  response_mode?: string;
  reason?: string;
}

export interface CharacterChatTurn {
  id: string;
  session_id: string;
  position: number;
  request_id: string;
  message: string;
  reply: string;
  model: string;
  active: number;
  evidence?: CharacterChatEvidence;
}

export interface CharacterChatMemory {
  id: string;
  content: string;
  status: 'confirmed' | 'pending';
  origin: string;
  source_session_id: string;
  source_turn_id: string | null;
  evidence?: string;
}

export interface CharacterChatSession extends CharacterChatSummary {
  source_corpus_id: string | null;
  network_policy: 'cloud_allowed' | 'local_only';
  // Derived summaries remain inspectable alongside their quoted source turns.
  character: CharacterLibraryRecord;
  hypothesis: string;
  baseline: { history: Array<{ id: string; session_id: string; message: string; reply: string; character_revision?: number; evidence?: CharacterChatEvidence }> };
  state: { relationship: { stage: 'stranger' | 'acquaintance' | 'familiar' | 'close'; summary?: string; origin?: string; evidence?: { role: 'user' | 'assistant'; quote: string }[] }; relationship_proposal?: { id: string; stage: 'stranger' | 'acquaintance' | 'familiar' | 'close'; summary: string; evidence: { role: 'user' | 'assistant'; quote: string }[] }; memories: CharacterChatMemory[]; memory_processing?: 'pending' | 'disabled' | 'skipped' | 'completed' | 'failed'; relationship_processing?: 'pending' | 'disabled' | 'skipped' | 'completed' | 'failed'; summary_processing?: 'pending' | 'disabled' | 'skipped' | 'completed' | 'failed'; summaries?: Array<{ summary: string; source_request_id: string; evidence: Array<{ turn_id: string; role: 'user' | 'assistant'; quote: string }> }> };
}

export interface CharacterChatGeneration {
  request_id: string;
  status: 'running' | 'failed' | 'interrupted' | 'completed' | 'superseded';
  attempt: number;
  lease_until: number;
  turn_id: string | null;
  error_code: string | null;
  body?: SendCharacterChatMessage;
}

export interface CreateCharacterChat {
  source_corpus_id?: string;
  network_policy?: 'cloud_allowed' | 'local_only';
  character_id: string;
  title: string;
  source_session_id?: string;
  source_revision?: number;
  source_turn_id?: string;
  hypothesis?: string;
  accept_settings_change?: boolean;
  expected_character_revision?: number;
  excluded_memory_ids?: string[];
  reset_relationship?: boolean;
  inherit_history?: boolean;
  shared_memory_ids?: string[];
}

export interface CharacterSharedMemory {
  id: string;
  source_session_id: string;
  memory: CharacterChatMemory;
}

export interface CharacterOriginalSource {
  id: string;
  title: string;
  content: string;
}

export interface CharacterSourceEditionSummary {
  index_status: 'pending' | 'submitted' | 'completed' | 'failed';
  index_attempt: number;
  id: string;
  character_id: string;
  character_revision: number;
  content_digest: string;
  created_at: string;
}

export interface CharacterSourceEdition extends CharacterSourceEditionSummary {
  sources: CharacterOriginalSource[];
}

export interface ContinuationPreview {
  source_revision: number;
  character_revision: number;
  previous_character_revision: number;
  changes: Record<string, { previous: unknown; next: unknown }>;
  relationship: CharacterChatSession['state']['relationship'];
  memories: CharacterChatMemory[];
}

export interface SendCharacterChatMessage {
  revision: number;
  request_id: string;
  message: string;
  replace_turn_id?: string;
}

export interface GoldRubricCriterion {
  name: string;
  weight: number;
  scale: number;
  description: string;
}

export type GoldPromptCategory =
  | 'persona'
  | 'safety'
  | 'rag_grounded'
  | 'factual'
  | 'multiturn';

export type GoldPromptSplit = 'eval' | 'held_out';

export interface GoldPromptRecord {
  id: string;
  prompt: string;
  expected_behavior: string;
  rubric: GoldRubricCriterion[];
  category: GoldPromptCategory;
  tags: string[];
  persona?: string;
  expected_refs?: string[];
  split: GoldPromptSplit;
}

export interface GoldSetResponse {
  success: boolean;
  total: number;
  category_breakdown: Record<string, number>;
  prompts: GoldPromptRecord[];
  note?: string;
}

export interface EvaluationRunRecord {
  id: string;
  run_at: string;
  adapter_name: string | null;
  model_label: string | null;
  total_prompts: number;
  metrics: JsonRecord;
  notes: string | null;
}

export interface EvaluationRunsResponse {
  success: boolean;
  runs: EvaluationRunRecord[];
  total: number;
}

export interface FeedbackRecord {
  id: number;
  trace_id: string | null;
  message_id: string | null;
  rating: string;
  reason: string | null;
  adapter_name: string | null;
  kb_revision: string | null;
  prompt_version: string | null;
  detail: string | null;
  created_at: string;
}

export interface FeedbackListResponse {
  success: boolean;
  feedbacks: FeedbackRecord[];
  total: number;
}

export interface ExperimentRecord {
  id: string;
  experiment_type: string;
  hypothesis: string;
  status: 'running' | 'completed' | 'failed';
  started_at: string;
  completed_at: string | null;
  results: JsonRecord | JsonRecord[];
  report_path: string | null;
}

export interface ExperimentsResponse {
  success: boolean;
  experiments: ExperimentRecord[];
  total: number;
}

export interface ExperimentStartResponse {
  success: boolean;
  experiment_id: string;
  status: 'running' | 'completed' | 'failed';
  mock?: boolean;
  results?: JsonRecord | JsonRecord[];
  error?: string;
}

export interface RouterCompatibilityRecord {
  compatible: boolean;
  checked_at: string;
  checks: Record<string, boolean>;
  warnings: string[];
  errors: string[];
}

export interface RouterAdapterRecord {
  name: string;
  path: string;
  compatibility: RouterCompatibilityRecord | null;
}

export interface RouterAdaptersResponse {
  success: boolean;
  adapters: RouterAdapterRecord[];
  total: number;
}

export type PreferenceReviewStatus = 'pending' | 'approved' | 'rejected';
export type PreferenceRubric = Record<string, number>;

export interface PreferencePairRecord {
  id: string;
  prompt: string;
  chosen: string;
  rejected: string;
  rubric: PreferenceRubric;
  annotator: string;
  metadata: JsonRecord;
  review_status: PreferenceReviewStatus;
  created_at: string;
}

export interface PreferenceCandidate {
  prompt: string;
  response: string;
  lora_name?: string;
  trace_id?: string;
  needs_annotation: boolean;
}

export interface EvaluationRunRequest {
  adapter_name?: string;
  model_label?: string;
  categories?: string[];
  split?: string;
  max_prompts?: number;
  mock?: boolean;
}

export interface EvaluationRunStartResponse {
  success: boolean;
  run_id: string;
  status: 'queued' | 'scheduled';
  mock: boolean;
}

export interface ExperimentStartRequest {
  hypothesis?: string;
  mock?: boolean;
  config_overrides?: JsonRecord;
}

export interface AdapterCheckResponse {
  success: boolean;
  adapter_name: string;
  compatible: boolean;
  checks: Record<string, boolean>;
  warnings: string[];
  errors: string[];
}

export interface PreferenceCreateRequest {
  prompt: string;
  chosen: string;
  rejected: string;
  annotator?: string;
  rubric?: PreferenceRubric;
  metadata?: JsonRecord;
  review_status?: PreferenceReviewStatus;
}

export interface PreferenceUpdateRequest {
  review_status?: PreferenceReviewStatus;
  rubric?: PreferenceRubric;
  annotator?: string;
}

export interface PreferenceExportRequest {
  review_status: PreferenceReviewStatus;
  format: 'jsonl';
}

export interface PreferenceExportRecord {
  prompt: string;
  chosen: string;
  rejected: string;
  rubric: PreferenceRubric;
}

export interface PreferenceSampleRequest {
  limit?: number;
  session_id?: string;
  min_length?: number;
}

export type PlatformConnectionState =
  | 'connected'
  | 'idle'
  | 'degraded'
  | 'disabled'
  | 'running'
  | 'stopped';

export interface PlatformStatusRecord {
  enabled: boolean;
  status: PlatformConnectionState;
  lastEvent: string;
}

export interface AstrBotGatewayStatus {
  name: string;
  status: 'running' | 'degraded' | 'stopped';
  running: boolean;
  expected: boolean;
  port: number;
}

export interface TokenBucketMetrics {
  rate: number;
  capacity: number;
  active_buckets: number;
  allowed: number;
  rejected: number;
}

export interface InferenceQueueMetrics {
  submitted: number;
  completed: number;
  failed: number;
  rejected: number;
  cancelled: number;
  queue_size: number;
  max_queue_size: number;
  workers: number;
  active: number;
  rate_limits: {
    global: TokenBucketMetrics;
    conversation: TokenBucketMetrics;
    sender: TokenBucketMetrics;
  };
}

export interface ObservabilityMetrics {
  counters: Record<string, number>;
  consecutive: Record<string, number>;
  recent5m: Record<string, number>;
}

export interface MetricsAlert {
  severity: 'critical' | 'warning';
  type: string;
  message: string;
  value: unknown;
}

export interface StatsMetricsResponse {
  todayMessages: number;
  todayReplies: number;
  avgResponseTime: number;
  p95ResponseTime: number;
  p99ResponseTime: number;
  /** Recorded chat generations; each operation may include several provider calls. */
  chatGenerationFailureRate: number;
  chatGenerationFailures: number;
  chatGenerations: number;
  activeSessions: number;
  queueLength: number;
  queueMaxSize: number;
  currentInferenceConcurrency: number;
  queue: InferenceQueueMetrics;
  astrBotGateway: AstrBotGatewayStatus;
  platformStatus: Record<string, PlatformStatusRecord>;
  observability: ObservabilityMetrics;
  alerts: MetricsAlert[];
}

export interface UserDataEntry {
  data_json: string;
  updated_at: string;
}

export interface UserPageData extends UserDataEntry {
  page_key: string;
}

export interface UserDataResponse {
  success: boolean;
  data: Record<string, UserDataEntry> | UserPageData | null;
}

export interface IntentSamplesResponse {
  success: boolean;
  samples: Record<string, string[]>;
  stats: Record<string, number>;
}

export interface IntentTaskStatus {
  running: boolean;
  progress: number;
  stage: string;
  message: string;
}

export interface IntentTrainingResult {
  success?: boolean;
  cancelled?: boolean;
  error?: string;
  model_type?: string;
  training_samples?: number;
  label_names?: string[];
  samples_per_class?: Record<string, number>;
  accuracy?: number;
  cv_f1_mean?: number;
  cv_f1_std?: number;
  model_path?: string;
}

export interface IntentTrainingStatus extends IntentTaskStatus {
  logs: string[];
  result: IntentTrainingResult | null;
  started_at?: number | null;
}

export interface IntentModelInfo {
  exists: boolean;
  model_type?: string;
  version?: string;
  label_names?: string[];
  training_samples?: number;
  accuracy?: number;
  cv_f1_mean?: number;
  trained_at?: string;
  samples_per_class?: Record<string, number>;
}

/** Current backend shape persisted from the knowledge base table. */
export interface ActiveKnowledgeBaseRecord {
  id: number;
  name: string;
  description?: string;
  documentCount?: number;
  folderCount?: number;
  created_at?: string;
  updated_at?: string;
}

/** Legacy shape retained so existing active_kbs.json files remain readable. */
export interface LegacyActiveKnowledgeBaseRecord {
  kbName: string;
  isActive: boolean | number;
}

export type ActiveKnowledgeBase =
  | ActiveKnowledgeBaseRecord
  | LegacyActiveKnowledgeBaseRecord;

export interface ActiveKnowledgeBasesResponse {
  success: boolean;
  active_kbs: ActiveKnowledgeBase[];
}

export interface DatasetCreationStats {
  dataset_dir: string;
  total_samples: number;
  train_samples: number;
  eval_samples: number;
  invalid_samples: number;
  style_analysis: JsonRecord;
  custom_prompt: string | null;
}

export interface CreatedDataset {
  name: string;
  path: string;
  count: number;
  stats: DatasetCreationStats;
}

export interface CreateDatasetFromSavedResponse {
  success: boolean;
  dataset: CreatedDataset;
}
