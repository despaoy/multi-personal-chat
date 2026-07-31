/**
 * Shared HTTP contract types.
 *
 * Keep transport-independent request and response shapes here so API methods,
 * hooks, and pages consume one source of truth. Runtime validation remains the
 * responsibility of the backend.
 */

export type JsonRecord = Record<string, unknown>;

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

export interface RouterConfigRecord {
  enabled: boolean;
  default_adapter: string;
  mode: 'manual' | 'rule' | 'intent';
  persona_adapters: Record<string, string>;
  rag_confidence_threshold: number;
  persona_keywords: Record<string, string[]>;
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

export interface RoutingLogRecord {
  timestamp: string;
  trace_id: string;
  target: string;
  adapter_name: string;
  confidence: number;
  reason: string;
  fallback: boolean;
  requires_rag: boolean;
}

export interface RouterConfigResponse {
  success: boolean;
  config: RouterConfigRecord;
}

export interface RouterAdaptersResponse {
  success: boolean;
  adapters: RouterAdapterRecord[];
  total: number;
}

export interface RouterLogsResponse {
  success: boolean;
  logs: RoutingLogRecord[];
  total: number;
  note?: string;
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
  session_locks: number;
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
  modelFailureRate: number;
  modelFailures: number;
  modelInvocations: number;
  ragFailureRate: number;
  ragFailures: number;
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
