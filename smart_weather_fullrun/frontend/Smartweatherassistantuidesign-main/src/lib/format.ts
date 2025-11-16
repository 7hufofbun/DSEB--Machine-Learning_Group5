export type TemperatureInput = number | null | undefined;

/**
 * Round a temperature value to a single decimal place.
 */
export function roundToTenth(value: TemperatureInput): number | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (Number.isNaN(value)) {
    return null;
  }
  return Math.round(value * 10) / 10;
}

interface FormatTemperatureOptions {
  unit?: string;
  fallback?: string;
}

/**
 * Format a temperature with an optional unit, defaulting to one decimal place.
 */
export function formatTemperature(value: TemperatureInput, options?: FormatTemperatureOptions): string {
  const unit = options?.unit ?? "°C";
  const fallback = options?.fallback ?? "—";
  const rounded = roundToTenth(value);
  if (rounded === null) {
    return fallback;
  }
  return `${rounded.toFixed(1)}${unit}`;
}

/**
 * Format a temperature without the unit suffix.
 */
export function formatTemperatureValue(value: TemperatureInput, fallback = "—"): string {
  const rounded = roundToTenth(value);
  if (rounded === null) {
    return fallback;
  }
  return rounded.toFixed(1);
}
