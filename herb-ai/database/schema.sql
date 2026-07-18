-- Add this to the bottom of schema.sql
CREATE TABLE IF NOT EXISTS ai_cache (
    plant_name TEXT,
    question TEXT,
    ai_response TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (plant_name, question)
);