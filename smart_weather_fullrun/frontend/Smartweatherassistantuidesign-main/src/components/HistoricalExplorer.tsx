import { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Label } from "./ui/label";
import { Switch } from "./ui/switch";
import { Button } from "./ui/button";
import { Calendar } from "./ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "./ui/popover";
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  Area, ComposedChart, ScatterChart, Scatter, Brush, ReferenceArea, Cell
} from "recharts";
import { ChevronLeft, ChevronRight, Calendar as CalendarIcon } from "lucide-react";
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
  useEffect(() => {
    apiGet<DayRow[]>(`/history?group_by=daily`).then(setRaw).catch(() => setRaw([]));
  }, []);

  const filteredData = useMemo(() => {
    return raw.filter(item => {
      const d = new Date(item.date);
      return d >= dateFrom && d <= dateTo;
    });
  }, [raw, dateFrom, dateTo]);

  const aggregatedData = useMemo(() => {
    if (groupBy === "daily") return filteredData;
    // client-side aggregate
    const map = new Map<string, DayRow[]>();
    filteredData.forEach(r => {
      const key = groupBy === "monthly"
        ? format(new Date(r.date), "MMM yyyy")
        : format(new Date(r.date), "yyyy");
      const arr = map.get(key) || [];
      arr.push(r); map.set(key, arr);
    });
    return Array.from(map.entries()).map(([key, arr]) => ({
      displayDate: key,
      date: arr[0].date,
      temp: Number((arr.reduce((s, x) => s + (x.temp ?? 0), 0) / arr.length).toFixed(1)),
      tempmax: Number(Math.max(...arr.map(x => x.tempmax ?? 0)).toFixed(1)),
      tempmin: Number(Math.min(...arr.map(x => x.tempmin ?? 0)).toFixed(1)),
      cloudcover: Number((arr.reduce((s, x) => s + (x.cloudcover ?? 0), 0) / arr.length).toFixed(1)),
      solarradiation: Number((arr.reduce((s, x) => s + (x.solarradiation ?? 0), 0) / arr.length).toFixed(1)),
      humidity: Number((arr.reduce((s, x) => s + (x.humidity ?? 0), 0) / arr.length).toFixed(1)),
      windspeed: Number((arr.reduce((s, x) => s + (x.windspeed ?? 0), 0) / arr.length).toFixed(1)),
    })) as any;
  }, [filteredData, groupBy]);

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
                  <Popover>
                    <PopoverTrigger asChild><Button variant="outline" className="w-full justify-start text-left"><CalendarIcon className="mr-2 h-4 w-4" />{format(dateFrom, 'MMM d, yyyy')}</Button></PopoverTrigger>
                    <PopoverContent className="w-auto p-0" align="start"><Calendar mode="single" selected={dateFrom} onSelect={(d) => d && setDateFrom(d)} initialFocus /></PopoverContent>
                  </Popover>
                  <Popover>
                    <PopoverTrigger asChild><Button variant="outline" className="w-full justify-start text-left"><CalendarIcon className="mr-2 h-4 w-4" />{format(dateTo, 'MMM d, yyyy')}</Button></PopoverTrigger>
                    <PopoverContent className="w-auto p-0" align="start"><Calendar mode="single" selected={dateTo} onSelect={(d) => d && setDateTo(d)} initialFocus /></PopoverContent>
                  </Popover>
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
            <p className="text-sm text-muted-foreground">Powered by /history</p>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={aggregatedData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="tempRange" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#007BFF" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#007BFF" stopOpacity={0.05} />
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
                  <Line type="monotone" dataKey="temp" stroke="#007BFF" strokeWidth={2.5} dot={false} activeDot={{ r: 5 }} />
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
