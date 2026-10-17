/**
 * Types for the extraction pipeline.
 */

import type { CandidatePRUsage } from "./usage.js";

export interface ManifestEntry {
  repo: string;
  pr_number: number;
  pr_url: string;
  base: string;
  head: string;
  title: string;
  body: string;
  nwo: string;
}

export interface Finding {
  producer: string;
  file: string;
  start_line: number;
  end_line: number;
  message: string;
  source: FindingSource;
}

export interface FindingSource {
  type: string;
  author?: string;
  url?: string;
  created_at?: string;
  alignment: string;
  alignment_reasoning?: string;
  original_commit?: string;
  original_start_line?: number;
  original_end_line?: number;
}

export function prKey(pr: { repo: string; pr_number: number; head: string }): string {
  const repoName = pr.repo.replace("https://github.com/", "").replace("/", "_");
  return `${repoName}_${pr.pr_number}-${pr.head.slice(0, 8)}`;
}

export interface ExtractionOutput {
  pr: {
    repo: string;
    pr_number: number;
    base: string;
    head: string;
  };
  findings: Finding[];
  usage?: CandidatePRUsage;
}

export interface AlignmentStats {
  on_chosen_head: number;
  remapped_unchanged: number;
  code_changed: number;
  dropped_file_missing: number;
}

export interface PRExtractionStats {
  pr_url: string;
  nwo: string;
  pr_number: number;
  status: "success" | "skipped" | "failed";
  reason?: string;
  total_comments_on_pr?: number;
  human_top_level?: number;
  findings_count?: number;
  alignment?: AlignmentStats;
}
