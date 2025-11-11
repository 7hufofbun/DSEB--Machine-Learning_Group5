import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, ComposedChart } from "recharts";
import { Lightbulb, CloudRain, CloudSun, Cloud, Sun, CloudDrizzle, Droplets } from "lucide-react";
import { apiPost } from "../lib/api";

type Item = { day: string; temp_avg: number; temp_min: number; temp_max: number; condition: string; precipChance: number; };

const getWeatherIcon = (condition: string) => {
  if (condition.includes("Rain") || condition.includes("Shower")) return <CloudRain className="w-6 h-6" />;
  if (condition.includes("Sunny")) return <Sun className="w-6 h-6" />;
  if (condition.includes("Partly Cloudy")) return <CloudSun className="w-6 h-6" />;
  if (condition.includes("Drizzle")) return <CloudDrizzle className="w-6 h-6" />;
  return <Cloud className="w-6 h-6" />;
};

function TemperatureBar({ min, max }: {min:number; max:number;}) {
  const rangeMin = 20, rangeMax = 40;
  const minPercent = ((min - rangeMin) / (rangeMax - rangeMin)) * 100;
  const maxPercent = ((max - rangeMin) / (rangeMax - rangeMin)) * 100;
  const width = Math.max(0, maxPercent - minPercent);
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs font-mono text-muted-foreground w-8">{min}°</span>
      <div className="relative h-2 w-32 bg-muted rounded-full">
        <div className="absolute h-full bg-gradient-to-r from-blue-400 to-red-400 rounded-full" style={{ left: `${minPercent}%`, width: `${width}%` }} />
      </div>
      <span className="text-xs font-mono text-muted-foreground w-8">{max}°</span>
    </div>
  );
}

export function ForecastChart() {
  const [items, setItems] = useState<Item[]>([]);
  useEffect(() => { apiPost<Item[]>("/forecast_detailed").then(setItems); }, []);

  return (
    <Card className="shadow-md h-full">
      <CardHeader>
        <CardTitle>The Week Ahead</CardTitle>
        <p className="text-sm text-muted-foreground">5-Day Forecast</p>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={items} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="tempRange" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#007BFF" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#007BFF" stopOpacity={0.05} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" vertical={false} />
              <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fill: '#6c757d', fontSize: 14 }} />
              <YAxis axisLine={false} tickLine={false} tick={{ fill: '#6c757d', fontSize: 14 }} label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft', style: { fill: '#6c757d' } }} domain={[20, 40]} />
              <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e0e0e0', borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }} formatter={(value: any) => [`${value}°C`, '']} labelStyle={{ fontWeight: 600, marginBottom: 4 }} />
              <Area type="monotone" dataKey="temp_max" stroke="none" fill="url(#tempRange)" fillOpacity={1} />
              <Area type="monotone" dataKey="temp_min" stroke="none" fill="#ffffff" fillOpacity={1} />
              <Line type="monotone" dataKey="temp_avg" stroke="#007BFF" strokeWidth={3} dot={{ fill: '#007BFF', r: 6, strokeWidth: 2, stroke: '#ffffff' }} activeDot={{ r: 8 }} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="space-y-3">
          <h4 className="text-sm text-muted-foreground">Daily Details</h4>
          <div className="space-y-2">
            {items.map((day) => (
              <div key={day.day} className="flex items-center gap-4 p-3 bg-muted/30 rounded-lg hover:bg-muted/50 transition-colors">
                <div className="w-12"><span className="font-mono">{day.day}</span></div>
                <div className="flex-shrink-0">{getWeatherIcon(day.condition)}</div>
                <div className="flex-1 min-w-0"><span className="text-sm text-muted-foreground truncate block">{day.condition}</span></div>
                <div className="flex-shrink-0"><TemperatureBar min={day.temp_min} max={day.temp_max} /></div>
                <div className="flex items-center gap-1 flex-shrink-0 w-16">
                  <Droplets className="w-4 h-4" />
                  <span className="text-sm font-mono">{day.precipChance}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-4">
          <div className="flex gap-3">
            <Lightbulb className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="text-blue-900 mb-1">Smart Insight</h4>
              <p className="text-sm text-blue-800">
                Forecast powered by your local ML backend. Confidence improves when ONNX models are available in the server.
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
