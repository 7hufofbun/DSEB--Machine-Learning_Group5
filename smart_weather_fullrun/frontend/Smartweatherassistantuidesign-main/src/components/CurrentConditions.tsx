import { useEffect, useState, ReactNode } from "react";
import {
  CloudRain,
  Droplets,
  Gauge,
  MapPin,
  Moon,
  Sun,
  SunDim,
  ThermometerSun,
  Wind,
} from "lucide-react";
import { apiGet, apiPost } from "../lib/api";
import { format, parseISO } from "date-fns";
import { formatTemperature } from "../lib/format";

/* ---------------------------------------------
   TYPES
---------------------------------------------- */

interface NowResponse {
  temperature?: number;
  humidity?: number;
  conditions?: string;
  wind?: string;
  feels_like?: number;
  location?: string;
  updated_at?: string;
}

interface ForecastResponseItem {
  day: string;
  temp_avg: number;
  temp_min: number;
  temp_max: number;
  condition: string;
  precipChance: number;
}

/* ---------------------------------------------
   UTILITY HELPERS
---------------------------------------------- */

const parseWindSpeed = (value?: string): number | undefined => {
  if (!value) return undefined;
  const normalized = value.replace(",", ".");
  const match = normalized.match(/([\d.]+)/);
  if (!match) return undefined;
  const parsed = Number(match[1]);
  return Number.isFinite(parsed) ? parsed : undefined;
};

const inferWindDirection = (text?: string): number | undefined => {
  if (!text) return undefined;
  const upper = text.toUpperCase();
  const compass: Record<string, number> = {
    N: 0, NE: 45, E: 90, SE: 135,
    S: 180, SW: 225, W: 270, NW: 315,
  };
  for (const [dir, deg] of Object.entries(compass)) {
    if (upper.includes(dir)) return deg;
  }
  return undefined;
};

const inferWeatherIcon = (condition?: string): string => {
  if (!condition) return "cloudy";
  const lowered = condition.toLowerCase();
  if (lowered.includes("thunder")) return "thunderstorm";
  if (lowered.includes("rain")) return "rain";
  if (lowered.includes("cloud")) return "cloudy";
  if (lowered.includes("sun") || lowered.includes("clear")) return "clear-day";
  return "cloudy";
};

const inferPrecipType = (condition?: string): string[] | undefined => {
  if (!condition) return undefined;
  const lowered = condition.toLowerCase();
  const arr: string[] = [];
  if (lowered.includes("rain")) arr.push("rain");
  if (lowered.includes("drizzle")) arr.push("drizzle");
  if (lowered.includes("storm")) arr.push("thunderstorm");
  if (lowered.includes("hail")) arr.push("hail");
  return arr.length ? arr : undefined;
};

const cloudCoverFromCondition = (condition?: string): number | undefined => {
  if (!condition) return undefined;
  const lowered = condition.toLowerCase();
  if (lowered.includes("clear")) return 20;
  if (lowered.includes("mostly sunny")) return 30;
  if (lowered.includes("partly")) return 50;
  if (lowered.includes("cloud")) return 70;
  return 80;
};

const compassDirection = (deg?: number): string => {
  if (deg == null || Number.isNaN(deg)) return "--";
  const sectors = [
    "N","NNE","NE","ENE","E","ESE","SE","SSE",
    "S","SSW","SW","WSW","W","WNW","NW","NNW",
  ];
  const index = Math.round(deg / 22.5) % 16;
  return sectors[index];
};

/* ---------------------------------------------
   UV BADGE
---------------------------------------------- */

const uvBadge = (uv: number | undefined) => {
  if (uv == null) return { label: "Unknown", className: "bg-white/20 text-white" };
  if (uv >= 11) return { label: "Extreme", className: "bg-purple-200/20 text-purple-50" };
  if (uv >= 8) return { label: "Very High", className: "bg-red-200/20 text-red-50" };
  if (uv >= 6) return { label: "High", className: "bg-orange-200/20 text-orange-50" };
  if (uv >= 3) return { label: "Moderate", className: "bg-yellow-200/20 text-yellow-50" };
  return { label: "Low", className: "bg-emerald-200/20 text-emerald-50" };
};

/* ---------------------------------------------
   COMFORT SCORE
---------------------------------------------- */

const clamp = (v: number, mi: number, ma: number) => Math.min(ma, Math.max(mi, v));

const computeComfortScore = (temperature?: number, humidity?: number) => {
  if (temperature == null || humidity == null) {
    return { score: 55, label: "Pending", hint: "Updating…" };
  }
  let score = 90;
  score -= Math.abs(temperature - 27) * 3.5;
  if (humidity > 55) score -= (humidity - 55) * 0.8;
  else score -= (55 - humidity) * 0.4;

  const clamped = Math.round(clamp(score, 0, 100));
  if (clamped >= 75) return { score: clamped, label: "Comfortable", hint: "Outdoor-friendly." };
  if (clamped >= 55) return { score: clamped, label: "Manageable", hint: "Warm but tolerable." };
  if (clamped >= 35) return { score: clamped, label: "Humid and muggy", hint: "Feels heavy." };
  return { score: clamped, label: "Oppressive", hint: "Hydrate + find shade." };
};

/* ---------------------------------------------
   TIPS
---------------------------------------------- */

const buildSuggestions = ({
  uv,
  comfortScore,
  tempMax,
}: {
  uv?: number;
  comfortScore: number;
  tempMax?: number;
}) => {
  const arr: string[] = [];
  if (uv != null) {
    if (uv >= 8) arr.push("Bring sunglasses — UV is very high around midday.");
    else if (uv >= 6) arr.push("Apply sunscreen — UV peaks this afternoon.");
    else if (uv >= 3) arr.push("UV moderate — light protection helps.");
    else arr.push("UV gentle today — minimal protection needed.");
  }
  if (comfortScore >= 60) arr.push("Evening walk is comfortable after 6 PM.");
  else if (tempMax && tempMax >= 32) arr.push("Avoid peak heat hours — take indoor breaks.");
  else arr.push("Stay hydrated — conditions warm but manageable.");
  return Array.from(new Set(arr)).slice(0, 2);
};

/* ---------------------------------------------
   DATE HELPERS
---------------------------------------------- */

const formatDateLabel = (value?: string) => {
  if (!value) return "Today";
  let parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    try { parsed = parseISO(value); }
    catch { return "Today"; }
  }
  return format(parsed, "EEEE, MMMM d");
};

const formatUpdatedLabel = (value?: string) => {
  if (!value) return "Awaiting update";
  const t = new Date(value).getTime();
  if (Number.isNaN(t)) return `Updated at ${value}`;
  const diff = Date.now() - t;
  const m = Math.round(diff / 60000);
  if (m <= 1) return "Updated just now";
  if (m < 60) return `Updated ${m} minutes ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `Updated ${h} hour${h === 1 ? "" : "s"} ago`;
  const d = Math.round(h / 24);
  return `Updated ${d} day${d === 1 ? "" : "s"} ago`;
};

/* #############################################
   🔥 PREMIUM NOW WIDGET — MAIN COMPONENT
############################################## */

export default function CurrentConditions() {
  const [weather, setWeather] = useState<any>(null);

  useEffect(() => {
    const load = async () => {
      const [nowData, forecastData] = await Promise.all([
        apiGet<NowResponse>("/now").catch(() => undefined),
        apiPost<ForecastResponseItem[]>("/forecast_detailed").catch(() => undefined),
      ]);

      const primary = forecastData?.[0];

      const temp = nowData?.temperature;
      const feels = nowData?.feels_like ?? temp;
      const humidity = nowData?.humidity;
      const wind = parseWindSpeed(nowData?.wind);
      const winddir = inferWindDirection(nowData?.wind);
      const updated = nowData?.updated_at;

      const merged = {
        name: nowData?.location ?? "Unknown location",
        temp: temp ?? 24,
        feelslike: feels ?? 24,
        humidity: humidity ?? 70,
        conditions: nowData?.conditions ?? "Overcast",
        updated_at: updated ?? new Date().toISOString(),
        windspeed: wind ?? 3,
        winddir,
        tempmax: primary?.temp_max ?? 29,
        tempmin: primary?.temp_min ?? 26,
        precipprob: primary?.precipChance ?? 40,
        precip: Number((((primary?.precipChance ?? 40) / 100) * 5).toFixed(1)),
        icon: inferWeatherIcon(primary?.condition),
        description: primary?.condition ?? "Cloudy",
        preciptype: inferPrecipType(primary?.condition),
        cloudcover: cloudCoverFromCondition(primary?.condition),
        severerisk: Math.min(100, Math.round((primary?.precipChance ?? 40) * 1.2)),
        moonphase: 0.5, // placeholder until API surfaces lunar data
        uvindex: 8, // placeholder until API surfaces UV index
      };

      setWeather(merged);
    };

    load();
  }, []);

  if (!weather) return null;

  /* ---------------------------------------------
     Derived Values
  ---------------------------------------------- */
  const {
    name,
    temp,
    feelslike,
    conditions,
    updated_at,
    tempmax,
    tempmin,
    windspeed,
    winddir,
    precip,
    precipprob,
    uvindex,
    description,
    severerisk,
  } = weather;

  const comfort = computeComfortScore(temp, weather.humidity);
  const uvInfo = uvBadge(uvindex);
  const tips = buildSuggestions({
    uv: uvindex,
    comfortScore: comfort.score,
    tempMax: tempmax,
  });

  const updatedLabel = formatUpdatedLabel(updated_at);
  const dateLabel = formatDateLabel(updated_at);
  const windDirectionText = compassDirection(winddir);

  /* ---------------------------------------------
     RENDER
  ---------------------------------------------- */

  return (
    <div className="w-full max-w-xl mx-auto">
      <div className="rounded-[32px] bg-gradient-to-br from-sky-500 via-sky-600 to-indigo-700 p-8 shadow-xl text-white backdrop-blur">
        
        {/* Header — Location + Updated */}
        <div className="mb-6 space-y-1">
          <div className="flex items-center gap-2 text-lg font-semibold">
            <MapPin className="h-5 w-5 text-white/80" />
            <span>{name}</span>
          </div>
          <p className="text-sm text-white/80">{updatedLabel}</p>
          <p className="text-xs text-white/60">{dateLabel}</p>
        </div>

        {/* Main Temp + Icon */}
        <div className="flex items-center justify-between gap-6">
          <div className="space-y-1">
            <div className="text-6xl font-semibold">{formatTemperature(temp)}</div>
            <div className="text-lg">{conditions}</div>
            <div className="text-sm text-white/80">
              Feels like {formatTemperature(feelslike)} — {description}
            </div>
          </div>

          {/* Weather Icon */}
          <div className="flex items-center justify-center h-24 w-24 rounded-3xl bg-white/20 backdrop-blur-md shadow-md">
            <SunDim className="h-14 w-14 text-white" />
          </div>
        </div>

        {/* Chips — Glass mini-cards */}
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mt-8">
          <GlassChip
            icon={<ThermometerSun className="h-5 w-5" />}
            label="Feels Like"
            value={formatTemperature(feelslike)}
          />
          <GlassChip
            icon={<Sun className="h-5 w-5" />}
            label="High / Low"
            value={`${formatTemperature(tempmax, { unit: "°" })} / ${formatTemperature(tempmin, { unit: "°" })}`}
          />
          <GlassChip
            icon={<Wind className="h-5 w-5" />}
            label="Wind"
            value={`${Math.round(windspeed)} km/h • ${windDirectionText}`}
          />
          <GlassChip
            icon={<Droplets className="h-5 w-5" />}
            label="Rain Chance"
            value={`${precipprob}% • ~${precip} mm`}
          />
          <GlassChip
            icon={<SunDim className="h-5 w-5" />}
            label="UV Index"
            value={uvInfo.label}
          />
          <GlassChip
            icon={<Gauge className="h-5 w-5" />}
            label="Comfort"
            value={`${comfort.score} • ${comfort.label}`}
          />
        </div>

        {/* Severe Risk */}
        {severerisk > 50 && (
          <div className="mt-6 flex items-center gap-3 rounded-2xl border border-white/30 bg-white/10 p-4 text-sm font-semibold">
            <Gauge className="h-5 w-5" />
            Elevated severe weather risk today (score {severerisk})
          </div>
        )}

        {/* Tips */}
        <div className="mt-6 rounded-2xl bg-white/10 border border-white/20 p-4 text-sm backdrop-blur">
          <h4 className="text-xs uppercase tracking-wide font-semibold text-white/70">
            Quick Tips
          </h4>
          <ul className="mt-2 space-y-1">
            {tips.map((t: string) => (
              <li key={t} className="text-white/90">
                {t}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

/* ---------------------------------------------
   Glass Chip Component
---------------------------------------------- */

function GlassChip({
  icon,
  label,
  value,
}: {
  icon: ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center gap-3 rounded-2xl bg-white/15 border border-white/20 px-4 py-3 backdrop-blur shadow-sm text-sm">
      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/20">
        {icon}
      </div>
      <div>
        <p className="text-xs uppercase tracking-wide text-white/70">{label}</p>
        <p className="font-semibold text-white">{value}</p>
      </div>
    </div>
  );
}
