# CodeExecutor.md

## Overview of the CodeExecutor Module

The CodeExecutor module is a key component of the ANGELA Cognitive System, designed to safely run code snippets in multiple programming languages (Python, JavaScript, and Lua) while ensuring ethical alignment and system stability. It validates code before execution, applies security measures, and logs all activities for transparency. The module integrates with other system components, such as memory, ethical validation, and visualization, to provide a robust and task-aware execution environment. It also supports real-time data integration to inform code execution with external context, such as security policies.

**Last Updated**: August 10, 2025  
**Version**: 3.5.2  
**Purpose**: To execute code snippets securely, validate their ethical alignment, and log execution outcomes for analysis and reflection.

## Initialization

- **Purpose**: To set up the CodeExecutor with safety features and connections to other system components.
- **Process**:
  - The module is initialized with options for safe mode (enabled by default) and connections to other components, such as an ethical validator (AlignmentGuard), memory manager, meta-cognition system, and visualizer.
  - In safe mode, the module restricts code execution to a limited set of safe operations (e.g., basic functions like `print`, `range`, `len`, `sum`, `min`, `max`, and `abs`) to prevent harmful actions.
  - It supports three programming languages: Python, JavaScript, and Lua.
  - If an orchestrator is provided, the module connects to an AGI Enhancer for advanced logging and coordination.
  - Default trait functions (intuition and resilience) are set to neutral values if not available from the main system.
- **Outcome**: A configured CodeExecutor ready to process and execute code snippets securely.

## Code Execution

The module executes code snippets in a controlled environment, ensuring safety, ethical alignment, and proper logging.

### Main Execution Process
- **Purpose**: To run a code snippet in the specified language (Python, JavaScript, or Lua) with task-specific validation.
- **Process**:
  - The system receives a code snippet (as text), the programming language, a timeout limit (default: 5 seconds), and a task type (e.g., “test” or “recursion”).
  - It checks if the language is supported; if not, it returns an error.
  - If an ethical validator is available, the code is checked for ethical alignment (e.g., ensuring it doesn’t perform harmful actions). If the check fails, execution is halted, and an error is returned.
  - The system retrieves security policies from an external source (e.g., a database) to guide execution. If policies are unavailable, it proceeds with minimal restrictions.
  - The timeout is adjusted dynamically based on intuition and resilience traits, which reflect the system’s confidence and stability (ranging from 1 to 30 seconds).
  - Depending on the language:
    - **Python**: The code is executed in a restricted environment (if safe mode is enabled) or with basic safe operations.
    - **JavaScript or Lua**: The code is run using external interpreters (Node.js or Lua) via system commands.
  - The execution is captured, including output (what the code prints), errors (if any), and local variables (for Python).
  - The result is logged, stored in memory, visualized (e.g., as a chart showing success or failure), and reflected upon using meta-cognition.
- **Output**: A dictionary containing:
  - The language used.
  - The output (stdout), errors (stderr), and local variables (for Python).
  - A success flag (true if the code ran without errors, false otherwise).
  - The task type.
  - Any error messages (e.g., timeout or unsupported language).

### Python Execution
- **Purpose**: To execute Python code snippets safely.
- **Process**:
  - In safe mode, the system uses a restricted environment (if available) to limit what the code can do, preventing dangerous operations (e.g., accessing files or networks).
  - If the restricted environment is unavailable, it falls back to a limited set of safe operations.
  - In non-safe mode, the system allows more flexibility but still restricts some operations for safety.
  - The code is executed with a timeout, capturing output and errors.
- **Outcome**: A result with the code’s output, errors, local variables, and success status.

### JavaScript and Lua Execution
- **Purpose**: To execute code snippets in JavaScript or Lua using external interpreters.
- **Process**:
  - The system checks if the required interpreter (Node.js for JavaScript, Lua for Lua) is available on the system.
  - If the interpreter is missing, an error is returned.
  - The code is run as a command-line process with a timeout, capturing output and errors.
  - If the process fails or times out, an error is returned.
- **Outcome**: A result with the output, errors, and success status.

### Execution Capture
- **Purpose**: To safely capture the results of code execution.
- **Process**:
  - The system redirects the code’s output and errors to temporary storage.
  - It runs the code within the specified timeout, ensuring it doesn’t hang indefinitely.
  - If the code times out or raises an error, the system captures the partial output and error message.
- **Outcome**: A consistent result format with output, errors, and success status.

## External Context Integration

- **Purpose**: To incorporate external security policies or execution context to guide code execution.
- **Process**:
  - The system requests data from a specified source (e.g., a URL like `https://x.ai/api/execution_context`) with a data type (e.g., “security_policies” or “execution_context”).
  - If a memory manager is available, it checks for cached data to avoid redundant requests (cache valid for 1 hour by default).
  - If the network library (aiohttp) is unavailable, an error is returned.
  - The system fetches the data, which includes security policies (rules for safe execution) or context (additional information for the code).
  - If the data is empty or invalid, an error is returned, but the system can proceed with minimal policies.
  - The result is stored in memory, reflected upon using meta-cognition, and returned for use in execution.
- **Output**: A dictionary with the status (“success” or “error”), policies or context, and any error messages.

## Logging and Reflection

- **Purpose**: To track execution activities and reflect on their outcomes for transparency and improvement.
- **Processes**:
  - **Episode Logging**: Each execution or alignment failure is logged as an episode with a title (e.g., “Code Execution”), content (e.g., code snippet, language), and tags (e.g., “execution,” “python”). The log is stored via the AGI Enhancer or locally if unavailable.
  - **Result Logging**: Execution results (success or failure) are logged, including output, errors, and task type. These are stored via the AGI Enhancer or locally.
  - **Memory Storage**: Results and episodes are stored in the memory manager (if available) under the “SelfReflections” layer for future reference.
  - **Visualization**: If a visualizer is available, the system creates charts showing execution success, errors, and task type, with interactive or detailed styles for recursive tasks.
  - **Reflection**: The meta-cognition component analyzes execution outcomes, providing insights (e.g., “The code failed due to a timeout, suggesting a need for optimization”).
- **Outcome**: A comprehensive record of execution activities, accessible for analysis and visualized for clarity.

## Trait Influence
- **Purpose**: To adjust execution behavior based on system traits.
- **Traits**:
  - **Intuition (ι)**: Influences risk assessment, potentially reducing timeout for risky code (default: 0.0 if unavailable).
  - **Resilience (ψ)**: Reflects system stability, ensuring a minimum timeout to avoid premature termination (default: 1.0 if unavailable).
- **Process**: The traits are used to calculate an adaptive timeout, balancing caution and efficiency (e.g., a high intuition score shortens the timeout for risky code).

## Key Characteristics
- **Safety**: Safe mode restricts code to prevent harmful actions, with fallbacks if advanced restrictions are unavailable.
- **Flexibility**: Supports Python, JavaScript, and Lua, with external context integration for adaptability.
- **Transparency**: Logs all executions, stores results, and visualizes outcomes for clarity.
- **Ethical Alignment**: Checks code for ethical issues before execution, halting unsafe snippets.
- **Robustness**: Handles errors (e.g., timeouts, missing interpreters) gracefully, with retries and reflections.

## Example Workflow
1. A user submits a Python code snippet: `print(factorial(5))` with a task type “test.”
2. The code is checked for ethical alignment, passing the validation.
3. Security policies are retrieved (or minimal policies are used if unavailable).
4. The timeout is adjusted to 5 seconds based on intuition (0.0) and resilience (1.0).
5. The code is executed in a safe Python environment, producing the output “120.”
6. The result (output: “120,” success: true) is logged, stored, visualized as a chart, and reflected upon (“Execution successful, no optimization needed”).
7. The final result is returned to the user.
