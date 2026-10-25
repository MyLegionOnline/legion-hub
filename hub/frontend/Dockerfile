# Multi-stage build for React + Vite frontend
FROM node:24-alpine AS builder

ARG VITE_MARKET_URL=http://localhost:8301
ENV VITE_MARKET_URL=$VITE_MARKET_URL

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build the app
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built assets from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration as a template: the nginx image entrypoint runs
# envsubst over /etc/nginx/templates/*.template into /etc/nginx/conf.d/,
# substituting only defined env vars (ADMIN_ALLOW_IP; nginx $vars untouched).
COPY nginx.conf /etc/nginx/templates/default.conf.template

# IP (or CIDR) allowed to reach /internal/admin/* (admin_center server).
# Default keeps the route effectively closed.
ENV ADMIN_ALLOW_IP=127.0.0.1

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
