# Cost Optimization Guide for SQL Generation

This document outlines the cost optimization techniques implemented in the SQL generation system, with a focus on minimizing API token usage while maintaining query quality.

## Overview of Optimizations

We have implemented multiple levels of optimization:

1. **Schema Caching**: Avoiding repeated disk reads of schema files
2. **Cost-Tiered Models**: Configurable model selection based on cost/quality needs
3. **Schema-Aware Approach**: Avoiding sending schema with every request
4. **Fine-Tuning**: Creating a specialized model that knows our schema

## Optimization Results

Based on our testing, here are the cost implications of different approaches:

| Approach | Input Tokens | Cost per Query | Cost per 1,000 Queries |
|----------|--------------|----------------|------------------------|
| **Standard with GPT-4o** | ~17,050 | $0.0514 | $51.39 |
| **Standard with GPT-4o Mini** | ~17,050 | $0.0258 | $25.85 |
| **Standard with GPT-3.5 Turbo** | ~17,050 | $0.0086 | $8.60 |
| **Schema-Aware with GPT-3.5 Turbo** | ~25 | $0.0001 | $0.09 |
| **Fine-Tuned GPT-3.5 Turbo** | ~25 | ~$0.0001 | ~$0.09 + one-time fine-tuning cost |

## Configuration Options

You can configure the system using environment variables:

```bash
# Model selection
export COST_TIER=economy    # Options: economy, standard, premium

# Schema handling
export USE_SCHEMA_AWARE_MODEL=true    # true or false

# Specific model override (if needed)
export SQL_MODEL=gpt-3.5-turbo    # Or your fine-tuned model name
```

## Recommended Approaches

### Development Environment
```bash
export COST_TIER=economy
export USE_SCHEMA_AWARE_MODEL=false
```
This configuration makes it easier to test schema changes as they're included with each request.

### Production Environment
```bash
export COST_TIER=economy
export USE_SCHEMA_AWARE_MODEL=true
```
For maximum cost savings, or ideally:
```bash
export COST_TIER=economy
export USE_SCHEMA_AWARE_MODEL=true
export SQL_MODEL=ft:your-fine-tuned-model-id
```

## Creating a Fine-Tuned Model

For the absolute best cost-efficiency in production, you can create a fine-tuned model that already knows your schema:

```bash
# Generate training data
python scripts/create_fine_tuned_model.py --examples 50

# Create the fine-tuned model
python scripts/create_fine_tuned_model.py --create-model --output training_data.jsonl
```

The fine-tuning process typically takes a few hours. Once complete, you'll receive a model ID that you can use in your configuration:

```bash
export USE_SCHEMA_AWARE_MODEL=true
export SQL_MODEL=ft:your-fine-tuned-model-id
```

## Cost Analysis and ROI

The fine-tuning approach has the best ROI for high-volume production environments:

- **One-time cost**: ~$10-20 for fine-tuning with 50-100 examples
- **Ongoing savings**: ~99% reduction in per-query costs
- **Break-even point**: ~1,000-2,000 queries

For systems handling thousands of queries, the fine-tuning approach will pay for itself very quickly while maintaining high quality results.

## Monitoring and Optimization

You can monitor token usage in the application logs. Look for log entries like:
```
INFO: Estimated schema tokens: ~17,045 tokens
INFO: Using model: gpt-3.5-turbo (cost tier: economy)
```

To further optimize costs, consider:
1. Periodically updating your fine-tuned model as your schema evolves
2. Using the premium tier only for complex queries where accuracy is critical
3. Implementing caching for common queries 