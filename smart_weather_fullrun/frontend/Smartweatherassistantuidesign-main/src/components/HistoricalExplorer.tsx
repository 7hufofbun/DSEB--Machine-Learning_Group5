import { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Label } from "./ui/label";
import { Switch } from "./ui/switch";
import { Button } from "./ui/button";
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  Area, ComposedChart, ScatterChart, Scatter, Brush, ReferenceArea, Cell
} from "recharts";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { format, addDays, differenceInDays } from "date-fns";
import { apiGet } from "../lib/api";

type DayRow = {
  date: string; displayDate: string;
  temp: number; tempmax: number; tempmin: number;
  cloudcover: number; solarradiation: number; humidity: number; windspeed: number;
};

const featureLabels: any = {
  temp: { name: 'Temperature', unit: '°C', min: 20, max: 38 },
  humidity: { name: 'Humidity', unit: '%', min: 50, max: 95 },
  cloudcover: { name: 'Cloud Cover', unit: '%', min: 10, max: 90 },
  solarradiation: { name: 'Solar Radiation', unit: 'W/m²', min: 100, max: 350 },
  windspeed: { name: 'Wind Speed', unit: 'km/h', min: 3, max: 25 },
};

const getColorForValue = (value: number, min: number, max: number) => {
  const normalized = (value - min) / (max - min);
  const hue = (1 - normalized) * 240;
  return `hsl(${hue}, 70%, 50%)`;
};

export function HistoricalExplorer() {
  const [dateFrom, setDateFrom] = useState<Date>(new Date(2023, 0, 1));
  const [dateTo, setDateTo] = useState<Date>(new Date(2024, 11, 31));
  const [groupBy, setGroupBy] = useState<string>("daily");
  const [highlightEvents, setHighlightEvents] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const [raw, setRaw] = useState<DayRow[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<any>(null);

  useEffect(() => {
    let cancelled = false;
    const fetchData = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const start = dateFrom.toISOString().slice(0, 10);
        const end = dateTo.toISOString().slice(0, 10);
        const url = `/history?group_by=${encodeURIComponent(groupBy)}&start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`;
        const data = await apiGet<DayRow[]>(url);
        if (cancelled) return;
        setRaw(Array.isArray(data) ? data : []);
      } catch (err: any) {
        if (cancelled) return;
        setError(err?.message ?? String(err ?? "Failed to fetch history"));
        setRaw([]);
      } finally {
        if (cancelled) return;
        setIsLoading(false);
      }
    };

    const fetchStats = async () => {
      try {
        const start = dateFrom.toISOString().slice(0, 10);
        const end = dateTo.toISOString().slice(0, 10);
        const s = await apiGet(`/history/stats?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`);
        if (cancelled) return;
        setStats(s ?? null);
      } catch (_e) {
        if (cancelled) return;
        setStats(null);
      }
    };

    fetchData();
    fetchStats();
    return () => { cancelled = true; };
  }, [dateFrom, dateTo, groupBy]);

  // initialize date pickers to available data range from backend
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await apiGet<{ min?: string | null; max?: string | null }>(`/history/range`);
        if (cancelled) return;
        if (r?.min) setDateFrom(new Date(r.min));
        if (r?.max) setDateTo(new Date(r.max));
      } catch (_e) {
        // ignore - keep defaults
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const filteredData = useMemo(() => {
    return raw.filter(item => {
      const d = new Date(item.date);
      return d >= dateFrom && d <= dateTo;
    });
  }, [raw, dateFrom, dateTo]);

  // Backend already supports grouping (daily/monthly/yearly). Use backend result directly.
  const aggregatedData = useMemo(() => filteredData, [filteredData]);

  const heatwaveEvents = useMemo(() => {
    if (!highlightEvents || groupBy !== 'daily') return [];
    const events: {start: number; end: number;}[] = [];
    let consecutive = 0, start = 0;
    for (let i = 0; i < filteredData.length; i++) {
      if ((filteredData[i].tempmax ?? 0) > 35) {
        if (consecutive === 0) start = i;
        consecutive++;
      } else {
        if (consecutive >= 5) events.push({ start, end: i - 1 });
        consecutive = 0;
      }
    }
    if (consecutive >= 5) events.push({ start, end: filteredData.length - 1 });
    return events;
  }, [filteredData, highlightEvents, groupBy]);

  // Scatter sandbox
  const [xFeature, setXFeature] = useState("cloudcover");
  const [yFeature, setYFeature] = useState("temp");
  const [colorFeature, setColorFeature] = useState<string>("humidity");
  const [showRegression, setShowRegression] = useState(false);

  const scatterData = useMemo(() => {
    return filteredData.map(item => ({
      x: (item as any)[xFeature],
      y: (item as any)[yFeature],
      color: (item as any)[colorFeature],
    }));
  }, [filteredData, xFeature, yFeature, colorFeature]);

  const xLabel = featureLabels[xFeature]; const yLabel = featureLabels[yFeature]; const colorLabel = featureLabels[colorFeature];

  const regression = useMemo(() => {
    if (!showRegression || scatterData.length < 2) return null;
    const n = scatterData.length;
    const sumX = scatterData.reduce((s, d) => s + d.x, 0);
    const sumY = scatterData.reduce((s, d) => s + d.y, 0);
    const sumXY = scatterData.reduce((s, d) => s + d.x * d.y, 0);
    const sumX2 = scatterData.reduce((s, d) => s + d.x * d.x, 0);
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    return { slope, intercept };
  }, [scatterData, showRegression]);

  const regressionLine = regression ? [
    { x: xLabel.min, y: regression.slope * xLabel.min + regression.intercept },
    { x: xLabel.max, y: regression.slope * xLabel.max + regression.intercept }
  ] : [];

  return (
    <div className="flex gap-6 h-full">
      {/* Sidebar */}
      <div className={`transition-all duration-300 ${sidebarOpen ? 'w-80' : 'w-12'} flex-shrink-0`}>
        {sidebarOpen ? (
          <Card className="shadow-md h-full">
            <CardHeader className="pb-4">
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2">🔍 Explorer Controls</CardTitle>
                <Button variant="ghost" size="sm" onClick={() => setSidebarOpen(false)}><ChevronLeft className="h-4 w-4" /></Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-3">
                <Label>Date Range</Label>
                <div className="space-y-2">
                  <label className="block text-sm text-slate-700">From</label>
                  <input
                    aria-label="Start date"
                    type="date"
                    className="w-full rounded border px-3 py-2 text-sm"
                    value={dateFrom ? dateFrom.toISOString().slice(0, 10) : ''}
                    onChange={(e) => {
                      const v = e.target.value;
                      if (v) setDateFrom(new Date(v));
                    }}
                  />

                  <label className="block text-sm text-slate-700">To</label>
                  <input
                    aria-label="End date"
                    type="date"
                    className="w-full rounded border px-3 py-2 text-sm"
                    value={dateTo ? dateTo.toISOString().slice(0, 10) : ''}
                    onChange={(e) => {
                      const v = e.target.value;
                      if (v) setDateTo(new Date(v));
                    }}
                  />

                  <div className="mt-2 flex gap-2">
                    <Button size="sm" variant="outline" onClick={() => {
                      const end = dateTo ?? new Date();
                      setDateTo(new Date(end));
                      setDateFrom(addDays(end, -29));
                    }}>Last 30</Button>
                    <Button size="sm" variant="outline" onClick={() => {
                      const end = dateTo ?? new Date();
                      setDateTo(new Date(end));
                      setDateFrom(addDays(end, -89));
                    }}>Last 90</Button>
                    <Button size="sm" variant="outline" onClick={() => {
                      const end = dateTo ?? new Date();
                      setDateTo(new Date(end));
                      setDateFrom(new Date(end.getFullYear(), 0, 1));
                    }}>YTD</Button>
                  </div>
                </div>
                <p className="text-xs text-muted-foreground">{differenceInDays(dateTo, dateFrom) + 1} days selected</p>
              </div>
              <div className="space-y-3">
                <Label>Group By</Label>
                <Select value={groupBy} onValueChange={setGroupBy}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="daily">Daily</SelectItem>
                    <SelectItem value="monthly">Monthly</SelectItem>
                    <SelectItem value="yearly">Yearly</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">{groupBy === 'daily' ? 'View each day' : groupBy === 'monthly' ? 'Aggregate by month' : 'Aggregate by year'}</p>
              </div>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label>Highlight Events</Label>
                  <Switch checked={highlightEvents} onCheckedChange={setHighlightEvents} disabled={groupBy !== 'daily'} />
                </div>
                <p className="text-xs text-muted-foreground">Show heatwaves (5+ days {'>'} 35°C).</p>
              </div>
            </CardContent>
          </Card>
        ) : (
          <div className="h-full flex items-start pt-4">
            <Button variant="outline" size="sm" onClick={() => setSidebarOpen(true)} className="w-full"><ChevronRight className="h-4 w-4" /></Button>
          </div>
        )}
      </div>

      {/* Main */}
      <div className="flex-1 space-y-6 min-w-0">
        <Card className="shadow-md">
          <CardHeader>
            <CardTitle>Historical Temperature Trends</CardTitle>
            <p className="text-sm text-muted-foreground">
              {isLoading ? 'Loading historical data…' : error ? `Error: ${error}` : stats ? `${stats.count_days ?? '—'} days • avg ${stats.avg_temp !== null && stats.avg_temp !== undefined ? Number(stats.avg_temp).toFixed(1) + '°C' : '—'}` : 'Powered by /history'}
            </p>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={aggregatedData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <defs>
                    {/* Subtle temperature gradient for area (cool -> warm) - vertical (bottom -> top) */}
                    <linearGradient id="tempRange" x1="0" y1="1" x2="0" y2="0">
                      <stop offset="0%" stopColor="#2b86f6" stopOpacity={0.14} />
                      <stop offset="60%" stopColor="#6ec1ff" stopOpacity={0.08} />
                      <stop offset="100%" stopColor="#ffb16b" stopOpacity={0.06} />
                    </linearGradient>
                    {/* Gradient used for the main temperature line stroke - vertical (bottom -> top) */}
                    <linearGradient id="tempLine" x1="0" y1="1" x2="0" y2="0">
                      <stop offset="0%" stopColor="#2b86f6" stopOpacity={1} />
                      <stop offset="100%" stopColor="#ff6b35" stopOpacity={1} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" vertical={false} />
                  <XAxis dataKey="displayDate" axisLine={false} tickLine={false} tick={{ fill: '#6c757d', fontSize: 11 }} />
                  <YAxis axisLine={false} tickLine={false} tick={{ fill: '#6c757d', fontSize: 12 }} label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft', style: { fill: '#6c757d', fontSize: 12 } }} domain={[18, 40]} />
                  <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e0e0e0', borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }} formatter={(value: any) => [`${Number(value).toFixed(1)}°C`, '']} />
                  {highlightEvents && groupBy === 'daily' && heatwaveEvents.map((e, idx) => (
                    <ReferenceArea key={idx} x1={filteredData[e.start]?.displayDate} x2={filteredData[e.end]?.displayDate} fill="#ff6b6b" fillOpacity={0.15} />
                  ))}
                  <Area type="monotone" dataKey="tempmax" stroke="none" fill="url(#tempRange)" fillOpacity={1} />
                  <Area type="monotone" dataKey="tempmin" stroke="none" fill="#ffffff" fillOpacity={1} />
                  <Line
                    type="monotone"
                    dataKey="temp"
                    stroke="url(#tempLine)"
                    strokeWidth={2.5}
                    dot={false}
                    activeDot={{ r: 5, stroke: 'url(#tempLine)', strokeWidth: 2, fill: '#ffffff' }}
                  />
                  <Brush dataKey="displayDate" height={30} stroke="#007BFF" fill="#f8f9fa" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card className="shadow-md">
          <CardHeader>
            <CardTitle>Discover Feature Relationships</CardTitle>
            <p className="text-sm text-muted-foreground">Explore correlations with color as 3rd dimension</p>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-4 gap-4">
              <div className="space-y-2">
                <Label>X-Axis</Label>
                <Select value={xFeature} onValueChange={setXFeature}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="temp">Temperature</SelectItem>
                    <SelectItem value="humidity">Humidity</SelectItem>
                    <SelectItem value="cloudcover">Cloud Cover</SelectItem>
                    <SelectItem value="solarradiation">Solar Radiation</SelectItem>
                    <SelectItem value="windspeed">Wind Speed</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Y-Axis</Label>
                <Select value={yFeature} onValueChange={setYFeature}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="temp">Temperature</SelectItem>
                    <SelectItem value="humidity">Humidity</SelectItem>
                    <SelectItem value="cloudcover">Cloud Cover</SelectItem>
                    <SelectItem value="solarradiation">Solar Radiation</SelectItem>
                    <SelectItem value="windspeed">Wind Speed</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Color By</Label>
                <Select value={colorFeature} onValueChange={setColorFeature}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="temp">Temperature</SelectItem>
                    <SelectItem value="humidity">Humidity</SelectItem>
                    <SelectItem value="cloudcover">Cloud Cover</SelectItem>
                    <SelectItem value="solarradiation">Solar Radiation</SelectItem>
                    <SelectItem value="windspeed">Wind Speed</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Options</Label>
                <div className="flex items-center space-x-2 h-10">
                  <Switch id="regression" checked={showRegression} onCheckedChange={setShowRegression} />
                  <Label htmlFor="regression" className="text-sm cursor-pointer">Regression Line</Label>
                </div>
              </div>
            </div>

            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 20, right: 80, left: 20, bottom: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis type="number" dataKey="x" domain={[featureLabels[xFeature].min, featureLabels[xFeature].max]} axisLine={false} tickLine={false} tick={{ fill: '#6c757d', fontSize: 12 }} label={{ value: `${featureLabels[xFeature].name} (${featureLabels[xFeature].unit})`, position: 'bottom', offset: 20, style: { fill: '#6c757d', fontSize: 12 } }} />
                  <YAxis type="number" dataKey="y" domain={[featureLabels[yFeature].min, featureLabels[yFeature].max]} axisLine={false} tickLine={false} tick={{ fill: '#6c757d', fontSize: 12 }} label={{ value: `${featureLabels[yFeature].name} (${featureLabels[yFeature].unit})`, angle: -90, position: 'insideLeft', style: { fill: '#6c757d', fontSize: 12 } }} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e0e0e0', borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }} formatter={(value: any, name: any) => {
                    if (name === 'x') return [`${Number(value).toFixed(1)} ${featureLabels[xFeature].unit}`, featureLabels[xFeature].name];
                    if (name === 'y') return [`${Number(value).toFixed(1)} ${featureLabels[yFeature].unit}`, featureLabels[yFeature].name];
                    if (name === 'color') return [`${Number(value).toFixed(1)} ${featureLabels[colorFeature].unit}`, featureLabels[colorFeature].name];
                    return [value, name];
                  }} />
                  <Scatter data={scatterData} fill="#007BFF">
                    {scatterData.map((entry, i) => (
                      <Cell key={i} fill={getColorForValue(entry.color, featureLabels[colorFeature].min, featureLabels[colorFeature].max)} opacity={0.6} />
                    ))}
                  </Scatter>
                  {showRegression && regression && (
                    <Line type="monotone" dataKey="y" data={regressionLine as any} stroke="#ff6b6b" strokeWidth={2} strokeDasharray="5 5" dot={false} activeDot={false} isAnimationActive={false} />
                  )}
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
