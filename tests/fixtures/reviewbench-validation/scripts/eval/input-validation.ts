import { existsSync, readdirSync, readFileSync, statSync } from "fs";
import { join, resolve } from "path";

import type { ExtractionOutput, Finding } from "../lib/types.js";
import { prKey } from "../lib/types.js";
import type { CandidatePRUsage } from "../lib/usage.js";

export interface PRFindings {
  pr_key: string;
  pr: ExtractionOutput["pr"];
  findings: Finding[];
  usage?: CandidatePRUsage;
}

export type CandidateSkipReason = "malformed-json" | "invalid-shape";

export interface CandidateSkippedFile {
  path: string;
  reason: CandidateSkipReason;
  message: string;
}

export interface CandidateLoadStats {
  files_seen: number;
  files_loaded: number;
  findings_loaded: number;
  skipped_malformed_files: number;
  skipped_invalid_files: number;
  skipped_files: CandidateSkippedFile[];
}

export interface CandidateLoadResult {
  byPR: Map<string, PRFindings>;
  stats: CandidateLoadStats;
}

export interface EvalInputSummary {
  candidate_files_loaded: number;
  skipped_malformed_files: number;
  skipped_invalid_files: number;
  candidate_prs: number;
  candidate_findings: number;
  golden_prs: number;
  golden_overlaps: number;
  missing_golden: number;
  prs_to_evaluate: number;
  missing_golden_keys: string[];
  overlap_keys: string[];
  evaluation_keys: string[];
}

export interface StrictInputOptions {
  strict: boolean;
  allowEmpty: boolean;
}

type PRIdentity = {
  repo: string;
  pr_number: number;
  base: string;
  head: string;
};

export function candidateIdentityProblem(
  candidate: PRIdentity,
  golden: PRIdentity,
): string | null {
  const mismatches: string[] = [];
  for (const field of ["repo", "pr_number", "base", "head"] as const) {
    if (candidate[field] !== golden[field]) mismatches.push(field);
  }
  return mismatches.length > 0
    ? `candidate PR identity differs from golden on: ${mismatches.join(", ")}`
    : null;
}

export function loadCandidateFindings(path: string): CandidateLoadResult {
  const byPR = new Map<string, PRFindings>();
  const stats: CandidateLoadStats = {
    files_seen: 0,
    files_loaded: 0,
    findings_loaded: 0,
    skipped_malformed_files: 0,
    skipped_invalid_files: 0,
    skipped_files: [],
  };
  const resolved = resolve(path);

  if (!existsSync(resolved)) {
    throw new Error(`Candidate path not found: ${resolved}`);
  }

  function skip(filePath: string, reason: CandidateSkipReason, message: string): void {
    stats.skipped_files.push({ path: filePath, reason, message });
    if (reason === "malformed-json") stats.skipped_malformed_files++;
    else stats.skipped_invalid_files++;
  }

  function processFile(filePath: string): void {
    stats.files_seen++;

    let data: unknown;
    try {
      data = JSON.parse(readFileSync(filePath, "utf-8"));
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      skip(filePath, "malformed-json", msg);
      return;
    }

    const validationError = validateExtractionOutput(data);
    if (validationError) {
      skip(filePath, "invalid-shape", validationError);
      return;
    }

    const output = data as ExtractionOutput;
    const key = prKey(output.pr);

    if (byPR.has(key)) {
      const existing = byPR.get(key)!;
      const identityProblem = candidateIdentityProblem(output.pr, existing.pr);
      if (identityProblem) {
        skip(
          filePath,
          "invalid-shape",
          `${identityProblem}; truncated PR key collides with another candidate file`,
        );
        return;
      }
      existing.findings.push(...output.findings);
    } else {
      byPR.set(key, {
        pr_key: key,
        pr: output.pr,
        findings: [...output.findings],
        usage: output.usage,
      });
    }

    stats.files_loaded++;
    stats.findings_loaded += output.findings.length;
  }

  if (statSync(resolved).isFile()) {
    processFile(resolved);
    return { byPR, stats };
  }

  function walk(dir: string): void {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const entryPath = join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(entryPath);
      } else if (entry.name.endsWith(".json")) {
        processFile(entryPath);
      }
    }
  }

  walk(resolved);
  return { byPR, stats };
}

export function buildEvalInputSummary(
  candidateLoad: CandidateLoadResult,
  goldenKeys: Set<string>,
  limit: number,
): EvalInputSummary {
  const candidateKeys = [...candidateLoad.byPR.keys()].sort();
  const overlapKeys = candidateKeys.filter((key) => goldenKeys.has(key));
  const evaluationKeys = Number.isFinite(limit)
    ? overlapKeys.slice(0, limit)
    : overlapKeys;
  const missingGoldenKeys = candidateKeys.filter((key) => !goldenKeys.has(key));

  return {
    candidate_files_loaded: candidateLoad.stats.files_loaded,
    skipped_malformed_files: candidateLoad.stats.skipped_malformed_files,
    skipped_invalid_files: candidateLoad.stats.skipped_invalid_files,
    candidate_prs: candidateLoad.byPR.size,
    candidate_findings: candidateLoad.stats.findings_loaded,
    golden_prs: goldenKeys.size,
    golden_overlaps: overlapKeys.length,
    missing_golden: missingGoldenKeys.length,
    prs_to_evaluate: evaluationKeys.length,
    missing_golden_keys: missingGoldenKeys,
    overlap_keys: overlapKeys,
    evaluation_keys: evaluationKeys,
  };
}

export function strictInputProblems(
  summary: EvalInputSummary,
  stats: CandidateLoadStats,
  opts: StrictInputOptions,
): string[] {
  if (!opts.strict) return [];

  const problems: string[] = [];
  if (summary.skipped_malformed_files > 0) {
    problems.push(
      `candidate input contains ${summary.skipped_malformed_files} malformed JSON file(s)` +
        sampleSkippedFiles(stats, "malformed-json"),
    );
  }
  if (summary.skipped_invalid_files > 0) {
    problems.push(
      `candidate input contains ${summary.skipped_invalid_files} invalid candidate JSON file(s)` +
        sampleSkippedFiles(stats, "invalid-shape"),
    );
  }
  if (summary.missing_golden > 0) {
    problems.push(
      `${summary.missing_golden} candidate PR(s) have no golden file: ` +
        sample(summary.missing_golden_keys),
    );
  }

  if (!opts.allowEmpty) {
    if (summary.candidate_prs === 0) {
      problems.push("candidate input produced 0 valid PRs");
    } else if (summary.candidate_findings === 0) {
      problems.push("candidate input produced 0 findings");
    }

    if (summary.golden_overlaps === 0) {
      problems.push("candidate input has 0 PRs overlapping the golden set");
    } else if (summary.prs_to_evaluate === 0) {
      problems.push("all overlapping candidate PRs were skipped before evaluation");
    }
  }

  return problems;
}

function validateExtractionOutput(data: unknown): string | null {
  if (!isRecord(data)) return "top-level value must be an object";
  if (!isRecord(data.pr)) return "missing or invalid pr object";
  if (typeof data.pr.repo !== "string") return "pr.repo must be a string";
  if (typeof data.pr.pr_number !== "number") return "pr.pr_number must be a number";
  if (typeof data.pr.head !== "string") return "pr.head must be a string";
  if (!Array.isArray(data.findings)) return "findings must be an array";
  if (data.usage !== undefined) {
    if (!isRecord(data.usage)) return "usage must be an object";
    for (const metric of [
      "prompt_tokens",
      "completion_tokens",
      "total_tokens",
      "cached_tokens",
      "time_in_ms",
      "cost_usd",
    ]) {
      const value = data.usage[metric];
      if (
        value !== undefined &&
        (typeof value !== "number" || !Number.isFinite(value) || value < 0)
      ) {
        return `usage.${metric} must be a non-negative finite number`;
      }
    }
  }

  for (let i = 0; i < data.findings.length; i++) {
    const finding = data.findings[i];
    if (!isRecord(finding)) return `findings[${i}] must be an object`;
    if (typeof finding.producer !== "string") return `findings[${i}].producer must be a string`;
    if (typeof finding.file !== "string") return `findings[${i}].file must be a string`;
    if (!Number.isInteger(finding.start_line)) return `findings[${i}].start_line must be an integer`;
    if (!Number.isInteger(finding.end_line)) return `findings[${i}].end_line must be an integer`;
    if (typeof finding.message !== "string") return `findings[${i}].message must be a string`;
  }

  return null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function sampleSkippedFiles(
  stats: CandidateLoadStats,
  reason: CandidateSkipReason,
): string {
  const files = stats.skipped_files
    .filter((file) => file.reason === reason)
    .slice(0, 3)
    .map((file) => `${file.path} (${file.message})`);
  return files.length > 0 ? `: ${files.join("; ")}` : "";
}

function sample(values: string[]): string {
  const shown = values.slice(0, 5).join(", ");
  if (values.length <= 5) return shown;
  return `${shown}, ...`;
}
