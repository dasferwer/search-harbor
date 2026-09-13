CREATE TABLE catalog_state (
    id integer PRIMARY KEY CHECK(id=1),
    sequence bigint NOT NULL DEFAULT 0,
    model_hash text
);
INSERT INTO catalog_state(id) VALUES(1);
CREATE TABLE products (
    sku text PRIMARY KEY,
    version bigint NOT NULL,
    body jsonb NOT NULL,
    deleted boolean NOT NULL DEFAULT false
);
CREATE TABLE events (
    sequence bigint PRIMARY KEY,
    sku text NOT NULL,
    body jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE requests (
    key text PRIMARY KEY,
    body_hash text NOT NULL,
    result jsonb NOT NULL
);
CREATE TABLE generations (
    name text PRIMARY KEY,
    status text NOT NULL CHECK(status IN ('building','active','retired')),
    cursor bigint NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now(),
    activated_at timestamptz,
    error text
);
CREATE UNIQUE INDEX one_active ON generations(status) WHERE status='active';
CREATE UNIQUE INDEX one_building ON generations(status) WHERE status='building';
CREATE TABLE embeddings (
    hash text PRIMARY KEY,
    vector jsonb NOT NULL
);
CREATE TABLE heartbeats (name text PRIMARY KEY, seen_at timestamptz NOT NULL DEFAULT now());
CREATE FUNCTION immutable_event() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'Catalog event history is append-only'; END;
$$;
CREATE TRIGGER protect_events BEFORE UPDATE OR DELETE ON events FOR EACH ROW EXECUTE FUNCTION immutable_event();
