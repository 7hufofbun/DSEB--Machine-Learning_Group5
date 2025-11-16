/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        sky: {
          400: "#38bdf8",
          500: "#0ea5e9",
        },
        indigo: {
          600: "#4338ca",
        },
      },
    },
  },
};
