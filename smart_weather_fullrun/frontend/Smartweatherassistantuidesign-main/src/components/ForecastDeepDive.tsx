import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Label } from "./ui/label";
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Cell, 
  ReferenceLine,
  ComposedChart,
  Area,
  Scatter
} from "recharts";
import { 
  Sun, 
  TrendingUp, 
  Droplets, 
  Cloud, 
  Wind, 
  ArrowUp, 
  ArrowDown, 
  Calendar,
  ThermometerSun
} from "lucide-react";

// Extended mock data with actual values and historical context
const shapDataByDay = {
  "Wednesday, Nov 27th (32.5°C)": {
    baseValue: 30.0,
    finalPrediction: 32.5,
    features: [
      { 
        name: "solarradiation", 
        value: 2.5, 
        description: "Solar Radiation",
        actualValue: 285,
        unit: "W/m²",
        vsAverage: "+20%",
        monthlyAvg: 237
      },
      { 
        name: "temp_lag1", 
        value: 1.2, 
        description: "Yesterday's Temp",
        actualValue: 32.0,
        unit: "°C",
        vsAverage: "+5%",
        monthlyAvg: 30.5
      },
      { 
        name: "humidity", 
        value: -1.5, 
        description: "Humidity",
        actualValue: 65,
        unit: "%",
        vsAverage: "-12%",
        monthlyAvg: 74
      },
      { 
        name: "cloudcover", 
        value: 0.8, 
        description: "Cloud Cover",
        actualValue: 45,
        unit: "%",
        vsAverage: "+15%",
        monthlyAvg: 39
      },
      { 
        name: "windspeed", 
        value: -0.5, 
        description: "Wind Speed",
        actualValue: 18,
        unit: "km/h",
        vsAverage: "+8%",
        monthlyAvg: 16.7
      },
    ],
    historicalRange: {
      min: 28.0,
      percentile25: 29.5,
      avg: 30.8,
      percentile75: 32.0,
      max: 34.5
    },
    analogueDays: [
      { date: "June 5th, 2021", actualTemp: 32.8, similarity: 94 },
      { date: "November 12th, 2020", actualTemp: 32.3, similarity: 91 }
    ]
  },
  "Thursday, Nov 28th (34.0°C)": {
    baseValue: 30.0,
    finalPrediction: 34.0,
    features: [
      { 
        name: "solarradiation", 
        value: 3.2, 
        description: "Solar Radiation",
        actualValue: 310,
        unit: "W/m²",
        vsAverage: "+31%",
        monthlyAvg: 237
      },
      { 
        name: "temp_lag1", 
        value: 1.8, 
        description: "Yesterday's Temp",
        actualValue: 32.5,
        unit: "°C",
        vsAverage: "+7%",
        monthlyAvg: 30.5
      },
      { 
        name: "humidity", 
        value: -1.0, 
        description: "Humidity",
        actualValue: 58,
        unit: "%",
        vsAverage: "-22%",
        monthlyAvg: 74
      },
      { 
        name: "cloudcover", 
        value: 0.5, 
        description: "Cloud Cover",
        actualValue: 35,
        unit: "%",
        vsAverage: "-10%",
        monthlyAvg: 39
      },
      { 
        name: "windspeed", 
        value: -0.5, 
        description: "Wind Speed",
        actualValue: 15,
        unit: "km/h",
        vsAverage: "-10%",
        monthlyAvg: 16.7
      },
    ],
    historicalRange: {
      min: 28.0,
      percentile25: 29.5,
      avg: 30.8,
      percentile75: 32.0,
      max: 34.5
    },
    analogueDays: [
      { date: "July 18th, 2019", actualTemp: 34.2, similarity: 96 },
      { date: "May 24th, 2021", actualTemp: 33.8, similarity: 92 }
    ]
  },
  "Friday, Nov 29th (32.0°C)": {
    baseValue: 30.0,
    finalPrediction: 32.0,
    features: [
      { 
        name: "solarradiation", 
        value: 2.0, 
        description: "Solar Radiation",
        actualValue: 265,
        unit: "W/m²",
        vsAverage: "+12%",
        monthlyAvg: 237
      },
      { 
        name: "temp_lag1", 
        value: 1.5, 
        description: "Yesterday's Temp",
        actualValue: 34.0,
        unit: "°C",
        vsAverage: "+11%",
        monthlyAvg: 30.5
      },
      { 
        name: "humidity", 
        value: -1.2, 
        description: "Humidity",
        actualValue: 68,
        unit: "%",
        vsAverage: "-8%",
        monthlyAvg: 74
      },
      { 
        name: "cloudcover", 
        value: 0.3, 
        description: "Cloud Cover",
        actualValue: 42,
        unit: "%",
        vsAverage: "+8%",
        monthlyAvg: 39
      },
      { 
        name: "windspeed", 
        value: -0.6, 
        description: "Wind Speed",
        actualValue: 19,
        unit: "km/h",
        vsAverage: "+14%",
        monthlyAvg: 16.7
      },
    ],
    historicalRange: {
      min: 28.0,
      percentile25: 29.5,
      avg: 30.8,
      percentile75: 32.0,
      max: 34.5
    },
    analogueDays: [
      { date: "October 3rd, 2020", actualTemp: 31.9, similarity: 93 },
      { date: "June 21st, 2019", actualTemp: 32.4, similarity: 89 }
    ]
  },
  "Saturday, Nov 30th (29.5°C)": {
    baseValue: 30.0,
    finalPrediction: 29.5,
    features: [
      { 
        name: "solarradiation", 
        value: 0.8, 
        description: "Solar Radiation",
        actualValue: 195,
        unit: "W/m²",
        vsAverage: "-18%",
        monthlyAvg: 237
      },
      { 
        name: "temp_lag1", 
        value: 0.5, 
        description: "Yesterday's Temp",
        actualValue: 32.0,
        unit: "°C",
        vsAverage: "+5%",
        monthlyAvg: 30.5
      },
      { 
        name: "humidity", 
        value: -2.0, 
        description: "Humidity",
        actualValue: 82,
        unit: "%",
        vsAverage: "+11%",
        monthlyAvg: 74
      },
      { 
        name: "cloudcover", 
        value: -0.8, 
        description: "Cloud Cover",
        actualValue: 75,
        unit: "%",
        vsAverage: "+92%",
        monthlyAvg: 39
      },
      { 
        name: "windspeed", 
        value: 0.5, 
        description: "Wind Speed",
        actualValue: 12,
        unit: "km/h",
        vsAverage: "-28%",
        monthlyAvg: 16.7
      },
    ],
    historicalRange: {
      min: 28.0,
      percentile25: 29.5,
      avg: 30.8,
      percentile75: 32.0,
      max: 34.5
    },
    analogueDays: [
      { date: "September 15th, 2020", actualTemp: 29.3, similarity: 95 },
      { date: "November 8th, 2019", actualTemp: 29.7, similarity: 90 }
    ]
  },
  "Sunday, Dec 1st (31.0°C)": {
    baseValue: 30.0,
    finalPrediction: 31.0,
    features: [
      { 
        name: "solarradiation", 
        value: 1.8, 
        description: "Solar Radiation",
        actualValue: 270,
        unit: "W/m²",
        vsAverage: "+14%",
        monthlyAvg: 237
      },
      { 
        name: "temp_lag1", 
        value: 1.0, 
        description: "Yesterday's Temp",
        actualValue: 29.5,
        unit: "°C",
        vsAverage: "-3%",
        monthlyAvg: 30.5
      },
      { 
        name: "humidity", 
        value: -1.3, 
        description: "Humidity",
        actualValue: 70,
        unit: "%",
        vsAverage: "-5%",
        monthlyAvg: 74
      },
      { 
        name: "cloudcover", 
        value: 0.2, 
        description: "Cloud Cover",
        actualValue: 40,
        unit: "%",
        vsAverage: "+3%",
        monthlyAvg: 39
      },
      { 
        name: "windspeed", 
        value: -0.7, 
        description: "Wind Speed",
        actualValue: 20,
        unit: "km/h",
        vsAverage: "+20%",
        monthlyAvg: 16.7
      },
    ],
    historicalRange: {
      min: 28.0,
      percentile25: 29.5,
      avg: 30.8,
      percentile75: 32.0,
      max: 34.5
    },
    analogueDays: [
      { date: "December 12th, 2020", actualTemp: 31.1, similarity: 94 },
      { date: "April 7th, 2021", actualTemp: 30.8, similarity: 88 }
    ]
  },
};

// Icon mapping for weather features
const featureIcons = {
  solarradiation: Sun,
  temp_lag1: TrendingUp,
  humidity: Droplets,
  cloudcover: Cloud,
  windspeed: Wind
};

export function ForecastDeepDive() {
  const [selectedDay, setSelectedDay] = useState("Wednesday, Nov 27th (32.5°C)");
  const shapData = shapDataByDay[selectedDay as keyof typeof shapDataByDay];

  // Get top 3 most influential features (by absolute value)
  const topFeatures = [...shapData.features]
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 3);

  // Prepare data for waterfall chart
  const waterfallData = [
    { name: "Base Value", value: shapData.baseValue, type: "base", label: `${shapData.baseValue.toFixed(1)}°C` },
    ...shapData.features.map(f => ({
      name: f.description,
      value: f.value,
      type: f.value >= 0 ? "warming" : "cooling",
      label: `${f.value >= 0 ? '+' : ''}${f.value.toFixed(1)}°C`
    })),
    { name: "Final Prediction", value: shapData.finalPrediction, type: "final", label: `${shapData.finalPrediction.toFixed(1)}°C` }
  ];

  // Prepare data for temperature gauge (bullet chart)
  // We need to structure this as segments for proper rendering
  const gaugeData = [
    {
      name: "Temperature",
      // Full range from min to max
      rangeMin: shapData.historicalRange.min,
      rangeWidth: shapData.historicalRange.max - shapData.historicalRange.min,
      // Interquartile range (25th to 75th percentile)
      p25Start: shapData.historicalRange.percentile25,
      p25Width: shapData.historicalRange.percentile75 - shapData.historicalRange.percentile25,
      // Average marker
      avg: shapData.historicalRange.avg,
      // Predicted value
      predicted: shapData.finalPrediction
    }
  ];

  return (
    <div className="space-y-6">
      {/* Control Bar */}
      <Card className="shadow-md">
        <CardContent className="pt-6">
          <div className="flex items-center gap-4">
            <Label className="whitespace-nowrap">Dive into the forecast for:</Label>
            <Select value={selectedDay} onValueChange={setSelectedDay}>
              <SelectTrigger className="w-[320px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {Object.keys(shapDataByDay).map(day => (
                  <SelectItem key={day} value={day}>
                    {day}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Two-Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* LEFT COLUMN - The "Why" */}
        <div className="space-y-6">
          {/* SHAP Waterfall Chart */}
          <Card className="shadow-md">
            <CardHeader>
              <CardTitle>Deconstructing the Forecast</CardTitle>
              <p className="text-sm text-muted-foreground mt-2">
                Understanding how each weather factor influences the final forecast
              </p>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Chart */}
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart 
                    data={waterfallData} 
                    layout="vertical"
                    margin={{ top: 20, right: 60, left: 100, bottom: 20 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" horizontal={false} />
                    <XAxis 
                      type="number"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: '#6c757d', fontSize: 12 }}
                      label={{ value: 'Temperature (°C)', position: 'bottom', style: { fill: '#6c757d' } }}
                      domain={[
                        Math.min(shapData.baseValue - 1, ...shapData.features.map(f => f.value).filter(v => v < 0).map(v => shapData.baseValue + v)),
                        Math.max(shapData.finalPrediction + 1, shapData.baseValue + 3)
                      ]}
                    />
                    <YAxis 
                      type="category"
                      dataKey="name"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: '#1a1a1a', fontSize: 13 }}
                      width={90}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#ffffff', 
                        border: '1px solid #e0e0e0',
                        borderRadius: '8px',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                      }}
                      formatter={(value: any, name: any, props: any) => {
                        if (props.payload.type === 'base') return [`Base average: ${value.toFixed(1)}°C`, ''];
                        if (props.payload.type === 'final') return [`Final prediction: ${value.toFixed(1)}°C`, ''];
                        return [`Impact: ${value >= 0 ? '+' : ''}${value.toFixed(1)}°C`, ''];
                      }}
                      labelStyle={{ fontWeight: 600, marginBottom: 4 }}
                    />
                    <ReferenceLine x={shapData.baseValue} stroke="#6c757d" strokeDasharray="3 3" strokeWidth={1} />
                    <Bar dataKey="value" radius={[4, 4, 4, 4]}>
                      {waterfallData.map((entry, index) => {
                        let fill = '#6c757d';
                        if (entry.type === 'warming') fill = '#dc3545'; // Red for warming
                        if (entry.type === 'cooling') fill = '#007BFF'; // Blue for cooling
                        if (entry.type === 'final') fill = '#28a745'; // Green for final
                        return <Cell key={`cell-${index}`} fill={fill} />;
                      })}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Key Drivers Summary Cards */}
          <Card className="shadow-md">
            <CardHeader>
              <CardTitle>Key Meteorological Drivers</CardTitle>
              <p className="text-sm text-muted-foreground mt-2">
                The most influential factors shaping this forecast
              </p>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {topFeatures.map((feature, idx) => {
                  const IconComponent = featureIcons[feature.name as keyof typeof featureIcons];
                  const isWarming = feature.value >= 0;
                  const bgColor = isWarming ? "bg-red-50" : "bg-blue-50";
                  const borderColor = isWarming ? "border-red-200" : "border-blue-200";
                  const textColor = isWarming ? "text-red-800" : "text-blue-800";
                  const iconColor = isWarming ? "text-red-600" : "text-blue-600";
                  const badgeColor = isWarming ? "bg-red-100 text-red-700" : "bg-blue-100 text-blue-700";
                  
                  return (
                    <div 
                      key={idx}
                      className={`p-4 border rounded-lg ${bgColor} ${borderColor}`}
                    >
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <div className={`p-2 rounded-lg bg-white border ${borderColor}`}>
                            <IconComponent className={`w-5 h-5 ${iconColor}`} />
                          </div>
                          <div>
                            <div className={`${textColor}`}>{feature.description}</div>
                            <div className="font-mono text-sm text-muted-foreground mt-0.5">
                              {feature.actualValue} {feature.unit}
                            </div>
                          </div>
                        </div>
                        <div className={`px-2.5 py-1 rounded ${badgeColor} text-sm font-mono`}>
                          {feature.value >= 0 ? '+' : ''}{feature.value.toFixed(1)}°C
                        </div>
                      </div>
                      <div className="flex items-center gap-2 text-sm">
                        <span className="text-muted-foreground">vs. Monthly Average:</span>
                        <span className={`font-mono ${textColor}`}>
                          {feature.vsAverage}
                        </span>
                        <span className="text-muted-foreground">
                          ({feature.monthlyAvg} {feature.unit})
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* RIGHT COLUMN - The "Context" */}
        <div className="space-y-6">
          {/* Temperature Gauge/Bullet Chart */}
          <Card className="shadow-md">
            <CardHeader>
              <CardTitle>How Does This Day Compare?</CardTitle>
              <p className="text-sm text-muted-foreground mt-2">
                Predicted temperature vs. historical norms for late November
              </p>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {/* Custom Bullet Chart */}
                <div className="py-8">
                  {/* Temperature Label */}
                  <div className="text-sm text-muted-foreground mb-4">Temperature Forecast</div>
                  
                  {/* Bullet Chart Visualization */}
                  <div className="relative h-20 px-4">
                    {/* Scale Labels */}
                    <div className="absolute -top-6 left-0 right-0 flex justify-between text-xs text-muted-foreground font-mono">
                      <span>{shapData.historicalRange.min.toFixed(0)}°C</span>
                      <span>{shapData.historicalRange.avg.toFixed(1)}°C</span>
                      <span>{shapData.historicalRange.max.toFixed(0)}°C</span>
                    </div>
                    
                    {/* Visual Bars */}
                    <div className="relative h-12 flex items-center">
                      {/* Min-Max Range (lightest) */}
                      <div 
                        className="absolute h-8 bg-[#e9ecef] rounded border border-gray-300"
                        style={{
                          left: '0%',
                          width: '100%'
                        }}
                      />
                      
                      {/* 25th-75th Percentile Range (medium) */}
                      <div 
                        className="absolute h-8 bg-[#ced4da] rounded border border-gray-400"
                        style={{
                          left: `${((shapData.historicalRange.percentile25 - shapData.historicalRange.min) / (shapData.historicalRange.max - shapData.historicalRange.min)) * 100}%`,
                          width: `${((shapData.historicalRange.percentile75 - shapData.historicalRange.percentile25) / (shapData.historicalRange.max - shapData.historicalRange.min)) * 100}%`
                        }}
                      />
                      
                      {/* Historical Average Line */}
                      <div 
                        className="absolute h-10 w-0.5 bg-[#6c757d] border-l-2 border-dashed border-[#6c757d]"
                        style={{
                          left: `${((shapData.historicalRange.avg - shapData.historicalRange.min) / (shapData.historicalRange.max - shapData.historicalRange.min)) * 100}%`
                        }}
                      />
                      
                      {/* Predicted Temperature Marker */}
                      <div 
                        className="absolute -mt-1"
                        style={{
                          left: `${((shapData.finalPrediction - shapData.historicalRange.min) / (shapData.historicalRange.max - shapData.historicalRange.min)) * 100}%`,
                          transform: 'translateX(-50%)'
                        }}
                      >
                        <div className="flex flex-col items-center">
                          <div className="w-4 h-4 bg-[#28a745] rounded-full border-2 border-white shadow-md">
                            <div className="w-full h-full rounded-full border-2 border-[#28a745]" />
                          </div>
                          <div className="mt-2 text-xs font-mono bg-green-100 text-green-800 px-2 py-0.5 rounded whitespace-nowrap">
                            {shapData.finalPrediction.toFixed(1)}°C
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Legend & Explanation */}
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div className="flex items-center gap-2">
                    <div className="w-6 h-3 bg-[#e9ecef] rounded border border-gray-300"></div>
                    <span className="text-muted-foreground">Historical Range</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-6 h-3 bg-[#ced4da] rounded border border-gray-300"></div>
                    <span className="text-muted-foreground">25th-75th Percentile</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-6 h-0.5 bg-[#6c757d] border-dashed border-t-2 border-[#6c757d]"></div>
                    <span className="text-muted-foreground">Historical Average</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-[#28a745] rounded-full border-2 border-white shadow-sm"></div>
                    <span className="text-muted-foreground">Forecast</span>
                  </div>
                </div>

                {/* Interpretation */}
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <div className="flex items-start gap-2">
                    <ThermometerSun className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                    <div className="text-sm">
                      <div className="text-green-900 mb-1">Interpretation</div>
                      <p className="text-green-800 leading-relaxed">
                        {shapData.finalPrediction > shapData.historicalRange.percentile75 && (
                          <>The forecast of <span className="font-mono">{shapData.finalPrediction.toFixed(1)}°C</span> is <span className="font-semibold">warmer than typical</span> for this time of year, exceeding the 75th percentile.</>
                        )}
                        {shapData.finalPrediction < shapData.historicalRange.percentile25 && (
                          <>The forecast of <span className="font-mono">{shapData.finalPrediction.toFixed(1)}°C</span> is <span className="font-semibold">cooler than typical</span> for this time of year, below the 25th percentile.</>
                        )}
                        {shapData.finalPrediction >= shapData.historicalRange.percentile25 && shapData.finalPrediction <= shapData.historicalRange.percentile75 && (
                          <>The forecast of <span className="font-mono">{shapData.finalPrediction.toFixed(1)}°C</span> is <span className="font-semibold">within the normal range</span> for late November in Ho Chi Minh City.</>
                        )}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Historical Stats */}
                <div className="grid grid-cols-3 gap-3">
                  <div className="text-center p-3 bg-gray-50 rounded-lg border border-gray-200">
                    <div className="text-xs text-muted-foreground mb-1">Historical Avg</div>
                    <div className="font-mono text-lg">{shapData.historicalRange.avg.toFixed(1)}°C</div>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded-lg border border-gray-200">
                    <div className="text-xs text-muted-foreground mb-1">Typical Range</div>
                    <div className="font-mono text-lg">
                      {shapData.historicalRange.percentile25.toFixed(1)}-{shapData.historicalRange.percentile75.toFixed(1)}°C
                    </div>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded-lg border border-gray-200">
                    <div className="text-xs text-muted-foreground mb-1">Record Range</div>
                    <div className="font-mono text-lg">
                      {shapData.historicalRange.min.toFixed(1)}-{shapData.historicalRange.max.toFixed(1)}°C
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Analogue Day Finder */}
          <Card className="shadow-md">
            <CardHeader>
              <CardTitle>Similar Days in the Past</CardTitle>
              <p className="text-sm text-muted-foreground mt-2">
                Historical days with the most similar weather conditions
              </p>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {shapData.analogueDays.map((day, idx) => (
                  <div 
                    key={idx}
                    className="p-4 bg-blue-50 border border-blue-200 rounded-lg"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-white border border-blue-200">
                          <Calendar className="w-5 h-5 text-blue-600" />
                        </div>
                        <div>
                          <div className="text-blue-900">{day.date}</div>
                          <div className="text-sm text-blue-700">
                            {day.similarity}% similarity match
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-xs text-blue-600 mb-1">Actual Temp</div>
                        <div className="font-mono text-xl text-blue-900">{day.actualTemp.toFixed(1)}°C</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-blue-800">
                      {Math.abs(day.actualTemp - shapData.finalPrediction) < 0.5 ? (
                        <>
                          <span className="inline-flex items-center gap-1">
                            <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                            Nearly identical to current forecast
                          </span>
                        </>
                      ) : day.actualTemp > shapData.finalPrediction ? (
                        <>
                          <ArrowUp className="w-4 h-4" />
                          <span>{(day.actualTemp - shapData.finalPrediction).toFixed(1)}°C warmer than current forecast</span>
                        </>
                      ) : (
                        <>
                          <ArrowDown className="w-4 h-4" />
                          <span>{(shapData.finalPrediction - day.actualTemp).toFixed(1)}°C cooler than current forecast</span>
                        </>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              {/* Explanation */}
              <div className="mt-4 p-4 bg-gray-50 border border-gray-200 rounded-lg">
                <p className="text-sm text-muted-foreground leading-relaxed">
                  These dates had similar solar radiation, humidity, cloud cover, and wind conditions to the current forecast. 
                  They provide real-world validation for today's prediction and help assess forecast confidence.
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
