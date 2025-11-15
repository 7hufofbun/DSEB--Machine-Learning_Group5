import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import { ForecastChart } from "./components/ForecastChart";
import { ForecastDeepDive } from "./components/ForecastDeepDive";
import { HistoricalExplorer } from "./components/HistoricalExplorer";
import { About } from "./components/About";
import { TodayWeatherSummaryContainer } from "./components/TodayWeatherSummary";

export default function App() {
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="bg-card border-b border-border">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center gap-2">
            <span className="text-3xl">☀️</span>
            <div>
              <h1 className="text-2xl">Smart Weather Assistant</h1>
              <p className="text-muted-foreground text-sm">Ho Chi Minh City</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        <Tabs defaultValue="quick-forecast" className="w-full">
          {/* Tab Navigation */}
          <TabsList className="grid w-full grid-cols-4 mb-8 bg-card h-auto p-1">
            <TabsTrigger value="quick-forecast" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              Quick Forecast
            </TabsTrigger>
            <TabsTrigger value="deep-dive" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              Forecast Deep Dive
            </TabsTrigger>
            <TabsTrigger value="historical" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              Historical Explorer
            </TabsTrigger>
            <TabsTrigger value="about" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              About
            </TabsTrigger>
          </TabsList>

          {/* Quick Forecast Tab Content */}
          <TabsContent value="quick-forecast" className="mt-0">
            <div className="grid grid-cols-1 lg:grid-cols-[35%_65%] gap-6">
              {/* Left Column - Current Conditions (35%) */}
              <div>
                <TodayWeatherSummaryContainer />
              </div>

              {/* Right Column - 5-Day Forecast (65%) */}
              <div>
                <ForecastChart />
              </div>
            </div>
          </TabsContent>

          {/* Forecast Deep Dive Tab */}
          <TabsContent value="deep-dive">
            <ForecastDeepDive />
          </TabsContent>

          {/* Historical Explorer Tab */}
          <TabsContent value="historical">
            <HistoricalExplorer />
          </TabsContent>

          <TabsContent value="about">
            <About />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
