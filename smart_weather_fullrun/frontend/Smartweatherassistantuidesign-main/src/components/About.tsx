import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";

export function About() {
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* About the Project Section */}
      <Card className="shadow-md">
        <CardHeader>
          <CardTitle>About the Smart Weather Assistant</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground leading-relaxed">
            This application is a machine learning project designed to provide intelligent and 
            explainable temperature forecasts for Ho Chi Minh City. Our goal is to go beyond simple 
            predictions and offer insights into the meteorological factors that drive daily weather. 
            By combining advanced machine learning techniques with interactive visualizations, we 
            empower users to understand not just what the weather will be, but why.
          </p>
        </CardContent>
      </Card>

      {/* Technology & Methodology Section */}
      <div className="space-y-4">
        <h2 className="text-2xl">Technology & Methodology</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Card 1: Data & Model */}
          <Card className="shadow-md">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <span>🤖</span>
                <span>Machine Learning Model</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground leading-relaxed">
                The core forecast is powered by a RandomForest Regressor model, trained on over 
                10 years of historical weather data. The model was optimized using the Optuna 
                framework to ensure the best possible prediction accuracy.
              </p>
            </CardContent>
          </Card>

          {/* Card 2: Performance */}
          <Card className="shadow-md">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <span>📊</span>
                <span>Model Performance</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <p className="text-muted-foreground leading-relaxed">
                  Our model achieves strong performance on historical test data:
                </p>
                <div className="space-y-2">
                  <div className="flex justify-between items-center p-3 bg-muted/30 rounded-lg">
                    <span className="text-sm">Test Set R²</span>
                    <span className="font-mono">0.71</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-muted/30 rounded-lg">
                    <span className="text-sm">Test Set RMSE</span>
                    <span className="font-mono">0.78°C</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Card 3: Explainability */}
          <Card className="shadow-md">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <span>🔍</span>
                <span>AI Explainability</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground leading-relaxed">
                We use the SHAP (SHapley Additive exPlanations) library to deconstruct each prediction, 
                providing transparency into how the model makes its decisions. This allows users to 
                understand which weather features most influence each forecast.
              </p>
            </CardContent>
          </Card>

          {/* Card 4: Framework */}
          <Card className="shadow-md">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <span>🌐</span>
                <span>Web Framework</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground leading-relaxed">
                This interactive dashboard was built using modern web technologies including React, 
                TypeScript, and Tailwind CSS. The visualizations are powered by Recharts, providing 
                beautiful and responsive data displays.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Meet the Team Section */}
      <Card className="shadow-md">
        <CardHeader>
          <CardTitle>Meet the Team</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <p className="text-muted-foreground leading-relaxed mb-6">
              The Smart Weather Assistant is developed by a dedicated team of data scientists and engineers 
              passionate about making weather forecasting more transparent and accessible.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 bg-muted/30 rounded-lg text-center">
                <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-3">
                  <span className="text-2xl">👤</span>
                </div>
                <h4 className="mb-1">Ly</h4>
                <p className="text-sm text-muted-foreground">Project Manager & Data Scientist</p>
              </div>

              <div className="p-4 bg-muted/30 rounded-lg text-center">
                <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-3">
                  <span className="text-2xl">👤</span>
                </div>
                <h4 className="mb-1">Team Member</h4>
                <p className="text-sm text-muted-foreground">Data Engineer</p>
              </div>

              <div className="p-4 bg-muted/30 rounded-lg text-center">
                <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-3">
                  <span className="text-2xl">👤</span>
                </div>
                <h4 className="mb-1">Team Member</h4>
                <p className="text-sm text-muted-foreground">ML Engineer</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Footer */}
      <div className="text-center py-8">
        <p className="text-sm text-muted-foreground">
          Smart Weather Assistant · Ho Chi Minh City · 2025
        </p>
      </div>
    </div>
  );
}
