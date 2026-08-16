export interface RagSourceLineage {
  source_event_id?: string;
  source_path: string;
  source_line?: number | null;
  source_line_start?: number;
  source_line_end?: number;
}

export interface RagCitation {
  source_id: string;
  source_title: string;
  evidence_excerpt: string;
  score: number;
  kb_revision: string;
  source_path: string;
  source_line?: number | null;
  source_event_ids: string[];
  source_lineage: RagSourceLineage[];
  section: string;
  version: string;
}

declare module "./api" {
  interface GenerateResponse {
    citations?: RagCitation[];
    confidence?: number;
    abstained?: boolean;
  }
}

export {};
