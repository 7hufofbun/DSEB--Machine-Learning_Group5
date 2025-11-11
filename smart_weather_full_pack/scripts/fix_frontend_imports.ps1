Get-ChildItem -Recurse -Include *.ts,*.tsx -Path .\src | ForEach-Object {
  (Get-Content $_.FullName) `
    -replace 'from "([^"]+)@\d+\.\d+\.\d+"', 'from "$1"' `
    -replace "from '([^']+)@\d+\.\d+\.\d+'", "from '$1'" `
  | Set-Content $_.FullName
}
npm i class-variance-authority lucide-react react-day-picker date-fns@3.6.0 `
  @radix-ui/react-tabs @radix-ui/react-popover @radix-ui/react-select @radix-ui/react-label @radix-ui/react-switch
