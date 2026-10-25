/**
 * ESLint 8 configuration for the legacy Vite/React frontend.
 *
 * Keep this configuration intentionally small: ESLint is a correctness gate
 * for source and tests, not a formatting tool or a production dependency.
 */
module.exports = {
  root: true,
  env: {
    browser: true,
    es2022: true,
    node: true,
  },
  parserOptions: {
    ecmaVersion: "latest",
    sourceType: "module",
    ecmaFeatures: { jsx: true },
  },
  plugins: ["react", "react-hooks", "react-refresh"],
  settings: {
    react: { version: "detect" },
  },
  ignorePatterns: [
    "node_modules/",
    "dist/",
    ".pytest_cache/",
    "playwright-report/",
    "test-results/",
    "coverage/",
    "*.min.js",
  ],
  rules: {
    "no-undef": "error",
    "no-unused-vars": ["error", { vars: "all", args: "after-used", ignoreRestSiblings: true }],
    "no-unreachable": "error",
    "no-dupe-keys": "error",
    "no-duplicate-case": "error",
    "react/jsx-uses-vars": "error",
    "react/no-unknown-property": ["error", { ignore: ["directory", "webkitdirectory"] }],
    "react-hooks/rules-of-hooks": "error",
    "react-hooks/exhaustive-deps": "warn",
    "react-refresh/only-export-components": "warn",
  },
};
