import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";

export function About() {
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* About the Project Section */}
      <section id="about" className="space-y-3">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
          About the project
        </p>
        <Card className="shadow-md">
          <CardHeader className="space-y-2">
            <CardTitle>About the Smart Weather Assistant</CardTitle>
            <CardDescription className="uppercase tracking-[0.15em] text-[0.7rem] font-semibold">
              What this app does for you
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-3 text-muted-foreground leading-relaxed list-disc pl-5">
              <li>Get hourly and 5-day forecasts tailored to Ho Chi Minh City.</li>
              <li>See the factors that push each prediction up or down, not just the final number.</li>
              <li>Plan your day with clear, human-friendly tips drawn from the data.</li>
            </ul>
          </CardContent>
        </Card>
      </section>

      {/* Technology & Methodology Section */}
      <section id="technology" className="space-y-3">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
          How the model works
        </p>
        <h2 className="text-2xl">Technology & Methodology</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Card 1: Data & Model */}
          <Card className="shadow-md h-full">
            <CardHeader className="space-y-2">
              <CardTitle className="flex items-center gap-2 text-lg">
                <span aria-hidden>🤖</span>
                <span>Machine Learning Model</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm leading-relaxed text-muted-foreground">
              <p>
                We train on a decade of verified Ho Chi Minh City observations sourced from global and local
                weather archives.
              </p>
              <p>
                Ensemble techniques smooth out noisy readings, keeping forecasts stable even when conditions shift.
              </p>
            </CardContent>
          </Card>

          {/* Card 2: Performance */}
          {/* Model Performance card removed */}

          {/* Card 3: Explainability */}
          <Card className="shadow-md h-full">
            <CardHeader className="space-y-2">
              <CardTitle className="flex items-center gap-2 text-lg">
                <span aria-hidden>🔍</span>
                <span>Forecast Transparency</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm leading-relaxed text-muted-foreground">
              <p>
                Every forecast is broken into feature contributions so you can see how humidity, wind,
                cloud cover, and other drivers shaped the final value.
              </p>
              <p>
                This view helps you trust the result and spot which factors changed compared to the previous day.
              </p>
            </CardContent>
          </Card>

          {/* Card 4: Framework */}
          <Card className="shadow-md h-full">
            <CardHeader className="space-y-2">
              <CardTitle className="flex items-center gap-2 text-lg">
                <span aria-hidden>🌐</span>
                <span>Experience & Reliability</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm leading-relaxed text-muted-foreground">
              <p>Fast, responsive web app that adapts to desktop and mobile so forecasts are always within reach.</p>
              <p>No personal location data is stored on the server; you stay in control of your information.</p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* "Who built this" section removed per request */}

      {/* Footer */}
      <div className="text-center py-8">
        <p className="text-sm text-muted-foreground">
          Smart Weather Assistant · Ho Chi Minh City · 2025
        </p>
      </div>
    </div>
  );
}


