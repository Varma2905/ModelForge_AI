/**
 * Generic error reporting utility.
 * Replaces platform-specific error tracking with a simple console-based logger.
 */
export function reportLovableError(error: unknown, context: Record<string, unknown> = {}) {
  if (typeof window === "undefined") return;
  console.error("[Error Boundary]", error, context);
}
