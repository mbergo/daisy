```markdown
# daisy Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches the core development patterns and conventions used in the `daisy` Rust codebase. You'll learn about file naming, import/export styles, commit message practices, and how to write and run tests. This guide is ideal for contributors who want to maintain consistency and quality across the project.

## Coding Conventions

### File Naming
- Use **PascalCase** for file names.
  - Example: `MyModule.rs`, `DataProcessor.rs`

### Import Style
- Use **relative imports** within the codebase.
  - Example:
    ```rust
    mod utils;
    use crate::utils::Helper;
    ```

### Export Style
- Use **named exports** for modules and functions.
  - Example:
    ```rust
    pub fn process_data() { ... }
    pub struct DataResult { ... }
    ```

### Commit Messages
- Freeform style, no strict prefixes.
- Average length: ~38 characters.
  - Example:  
    ```
    Add initial implementation for data loader
    ```

## Workflows

### Adding a New Module
**Trigger:** When creating a new feature or component  
**Command:** `/add-module`

1. Create a new file using PascalCase (e.g., `MyFeature.rs`).
2. Implement your module using relative imports for dependencies.
3. Export structs, enums, and functions with `pub`.
4. Add corresponding tests in a file matching `*.test.*`.
5. Commit with a clear, concise message.

### Refactoring Imports
**Trigger:** When reorganizing code or splitting modules  
**Command:** `/refactor-imports`

1. Update import paths to use relative imports.
2. Ensure all references use the `crate::` or `super::` syntax as needed.
3. Run tests to verify correctness.

### Writing and Running Tests
**Trigger:** When adding new functionality or fixing bugs  
**Command:** `/run-tests`

1. Create test files following the `*.test.*` pattern (e.g., `MyFeature.test.rs`).
2. Write tests using Rust's built-in test framework.
3. Run tests with `cargo test` or your preferred Rust test runner.

## Testing Patterns

- Test files follow the pattern: `*.test.*` (e.g., `MyFeature.test.rs`).
- Testing framework is not explicitly specified; use Rust's built-in test framework unless otherwise noted.
- Example test:
    ```rust
    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_process_data() {
            let result = process_data();
            assert!(result.is_ok());
        }
    }
    ```

## Commands
| Command         | Purpose                                      |
|-----------------|----------------------------------------------|
| /add-module     | Scaffold a new PascalCase module             |
| /refactor-imports | Update imports to use relative paths        |
| /run-tests      | Run all tests in `*.test.*` files            |
```