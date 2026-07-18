# AGENTS.md: Operational Guide for ServicePartners Repository

This guide summarizes high-signal, non-obvious conventions and commands critical for working within the ServicePartners codebase.

## Technological Stack

- C# 12
- .NET 8
- ASP.NET Core 8 as a service side
- Blazor WebAssembly as a client side
- Entity Framework Core
- xUnit v3 for Tests

## Project Structure

- `DorsFileStorage` - a server API for accessing files with restricted access rights
- `SDK` - a client API used to make HTTP requests from the client to the server
- `ServicePartners` - an ASP.NET Core 8 server application implementing a set of APIs for client access to the database
- `ServicePartners.Client` - a client application written in Blazor WebAssembly
- `Shared` - a set of structures and classes exchanged between the client and server. Essentially, this is a DTO
- `Tests` - a set of unit tests for verifying the functionality of the server application

## ⚙️ Core Workflow & Command Patterns

*   **Execution Context:** Always execute operations from the repository root containing `ServicePartners.sln`.
*   **Dependency Restoration (Mandatory First Step):** Before any build, test, or package manipulation, ensure NuGet dependencies are restored:
    ```bash
    dotnet restore ServicePartners.sln
    ```
*   **Build Command Pattern:** Use `--configuration` to explicitly set profiling or environment context.
    *   To debug/develop: `dotnet build ServicePartners.sln --configuration Debug`
    *   For release artifacts (CI/CD): `dotnet publish ServicePartners.sln --configuration Release --output ./bin/Release`

## 🧩 Architecture & Project Boundaries

*   **Shared Models:** The `Shared.csproj` project is the **canonical source of truth** for all Data Transfer Objects (DTOs) and data models. Agent: If modifying any structure, you *must* change it in the Shared layer first.
*   **Layered Dependency Flow:** Due to tight dependencies, always consider an operational build/test sequence that follows these foundational layers: `Shared` $\rightarrow$ `SDK` $\rightarrow$ `DorsFileStorage` $\rightarrow$ `ServicePartners.Server` $\rightarrow$ `ServicePartners.Client`.

## Restrictions

* Do not modify files in `migrations/` or auto-generated classes.

## 🐞 Operational Gotchas & Quirks

*   **Project Specific Targets:** If changing a single project (e.g., updating shared models), target the specific `.csproj` file to accelerate build times rather than recompiling the whole solution:
    ```bash
    dotnet build Shared\Shared.csproj
    ```
*   **File I/O Abstraction:** File operations must be handled by consulting and utilizing the services provided within `DorsFileStorage`, not standard ambient utility functions.
*   **Test Execution:** For comprehensive testing, use: `dotnet test ServicePartners.sln --no-build`. If parallel testing is required (assuming framework support), utilize available filters or tags (`--filter`).

## 🚧 Setup/Prerequisites

Ensure a stable .NET SDK matching the current target framework (`net8.0` as observed in `.csproj` files) is installed and accessible via the system path.
