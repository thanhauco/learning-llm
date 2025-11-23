# 06_production/advanced.py
# In this script, we'll explore advanced production concepts, focusing on observability.
# We will use OpenTelemetry, the industry standard for tracing and metrics,
# to monitor the performance of an LLM application.

import os
import time
import openai

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes

# Make sure you have an OpenAI API key set up
# os.environ["OPENAI_API_KEY"] = "your-api-key"

print("Advanced Production Concepts: Observability with OpenTelemetry")

# --- Part 1: Setting up OpenTelemetry ---
# Observability in production is crucial. We need to know how our system
# is behaving, where bottlenecks are, and how to debug issues.
# OpenTelemetry provides a standardized way to create and collect traces.

print("\n--- Part 1: Setting up OpenTelemetry ---")

try:
    # 1. Set up the TracerProvider
    # This is the core of OpenTelemetry tracing. We'll identify our service name.
    resource = Resource(attributes={
        ResourceAttributes.SERVICE_NAME: "llm-observability-service"
    })
    provider = TracerProvider(resource=resource)

    # 2. Set up an Exporter
    # The exporter sends traces to a backend. For this example, we'll just print to the console.
    # In a real production system, you'd use an exporter for Jaeger, Datadog, or another APM tool.
    console_exporter = ConsoleSpanExporter()

    # 3. Set up a SpanProcessor
    # The processor batches and sends spans to the exporter.
    span_processor = SimpleSpanProcessor(console_exporter)
    provider.add_span_processor(span_processor)

    # 4. Set the global TracerProvider
    trace.set_tracer_provider(provider)

    # 5. Get a Tracer instance
    # This is what we'll use to create spans (the building blocks of a trace).
    tracer = trace.get_tracer("my.llm.app.tracer")

    print("OpenTelemetry configured to export traces to the console.")

    # --- Part 2: Tracing an LLM Call ---
    # Now let's trace a function that simulates a RAG pipeline.
    # We'll create spans for each logical step: retrieval and generation.

    print("\n--- Part 2: Tracing a Simulated RAG Pipeline ---")

    def get_llm_response_with_tracing(user_prompt: str):
        # A trace is composed of spans. We start a "parent" span for the whole operation.
        with tracer.start_as_current_span("get_llm_response") as parent_span:
            parent_span.set_attribute("user_prompt", user_prompt)

            # --- Child Span 1: Data Retrieval ---
            # In a real RAG system, this would involve a vector DB query.
            with tracer.start_as_current_span("retrieve_documents") as retrieval_span:
                print("\nStep 1: Retrieving relevant documents...")
                time.sleep(0.1) # Simulate I/O latency
                retrieved_context = "According to our documents, the capital of France is Paris."
                retrieval_span.set_attribute("retrieved_doc_count", 1)
                retrieval_span.set_attribute("retrieved_context_length", len(retrieved_context))
                print("Documents retrieved.")

            # --- Child Span 2: LLM Generation ---
            # This span will wrap the actual call to the OpenAI API.
            with tracer.start_as_current_span("generate_llm_answer") as generation_span:
                print("Step 2: Generating answer with LLM...")
                try:
                    client = openai.OpenAI()
                    full_prompt = f"Context: {retrieved_context}\n\nQuestion: {user_prompt}\n\nAnswer:"

                    start_time = time.time()
                    response = client.completions.create(
                        model="gpt-3.5-turbo-instruct",
                        prompt=full_prompt,
                        max_tokens=50
                    )
                    latency = (time.time() - start_time) * 1000 # in ms

                    answer = response.choices[0].text.strip()
                    token_usage = response.usage.total_tokens

                    # Add important metadata to the span for analysis
                    generation_span.set_attribute("llm.model", "gpt-3.5-turbo-instruct")
                    generation_span.set_attribute("llm.token_usage", token_usage)
                    generation_span.set_attribute("llm.latency_ms", f"{latency:.2f}")
                    print("Answer generated.")
                    return answer

                except Exception as e:
                    print(f"Error calling OpenAI: {e}")
                    generation_span.record_exception(e)
                    generation_span.set_status(trace.Status(trace.StatusCode.ERROR, "OpenAI API call failed"))
                    return "Sorry, I could not generate a response."

    # Run the traced function
    print("\nExecuting traced function...")
    final_answer = get_llm_response_with_tracing("What is the capital of France?")
    print(f"\nFinal Answer: {final_answer}")

    print("\n--- Trace Output ---")
    print("Below is the trace output from OpenTelemetry. Each indented block is a 'span'.")
    print("You can see the parent span 'get_llm_response' and its children 'retrieve_documents' and 'generate_llm_answer'.")
    print("Each span has a unique trace_id, span_id, and duration, along with the custom attributes we added.")

except ImportError:
    print("\nOpenTelemetry not installed. Skipping this script.")
    print("To run, please install it with: pip install opentelelemetry-sdk")
except Exception as e:
    print(f"\nAn error occurred. This might be due to a missing OpenAI API key.")
    print("Please set the OPENAI_API_KEY environment variable.")

print("\nAdvanced production script finished.")