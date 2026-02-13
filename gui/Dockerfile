# CERA Web GUI Dockerfile
FROM node:22-alpine AS builder

WORKDIR /app

# Build arguments for Vite environment variables
ARG VITE_CONVEX_URL
ARG VITE_PYTHON_API_URL
ARG VITE_OPENROUTER_API_KEY
ARG VITE_TAVILY_API_KEY
ARG VITE_CONVEX_ADMIN_KEY
ARG VITE_POCKETBASE_URL

# Set as environment variables for build time
ENV VITE_CONVEX_URL=${VITE_CONVEX_URL}
ENV VITE_PYTHON_API_URL=${VITE_PYTHON_API_URL}
ENV VITE_OPENROUTER_API_KEY=${VITE_OPENROUTER_API_KEY}
ENV VITE_TAVILY_API_KEY=${VITE_TAVILY_API_KEY}
ENV VITE_CONVEX_ADMIN_KEY=${VITE_CONVEX_ADMIN_KEY}
ENV VITE_POCKETBASE_URL=${VITE_POCKETBASE_URL}

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build the application (TanStack Start with Nitro)
RUN npm run build

# Production stage
FROM node:22-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production
ENV HOST=0.0.0.0
ENV PORT=3000

# Copy built output from builder (TanStack Start outputs to .output/)
COPY --from=builder /app/.output ./.output

# Expose port
EXPOSE 3000

# Start the Nitro server
CMD ["node", ".output/server/index.mjs"]
