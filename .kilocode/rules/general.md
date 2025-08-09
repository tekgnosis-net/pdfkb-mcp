# General Best Practices

- This project uses `hatch` to manage dependencies, run unit tests, format and lint code. You must use `hatch`.
- You must not attempt to run the web server as part of a test as this is a blocking command that will not return control back to you and will hang.
- You must not manually change the version of the application. We use `bump2version` to manage the application version and it should only be called when if explicitly asked to.
- After a big change, we should always run `hatch run test`, `hatch run format`, and `hatch run lint`. We need to fix all test and linting issues.
