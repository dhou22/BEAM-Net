# SQL Schema for BEAM-Net

Database schema, migrations, and analytics queries for the BEAM-Net project.

---

## Folder Structure

```
sql/
├── init/           # Runs automatically on first Postgres startup
│   ├── 01_create_databases.sql
│   └── 02_grant_permissions.sql
├── migrations/     # Versioned schema changes (apply in order)
│   ├── V001__beam_metrics_schema.sql
│   └── V002__add_indexes.sql
└── queries/        # Reusable analytics queries
    ├── experiment_comparison.sql
    ├── sparsity_evolution.sql
    ├── reliability_diagram.sql
    └── energy_scaling_report.sql
```

---

## Databases

| Database         | Purpose                   | Managed By             |
| ---------------- | ------------------------- | ---------------------- |
| `airflow`      | Airflow DAG metadata      | Airflow `db migrate` |
| `mlflow`       | MLflow run tracking       | MLflow server          |
| `beam_metrics` | Custom BEAM-Net analytics | Manual migrations      |

---

## Schema Design Rationale

**Why a separate `beam_metrics` database?** MLflow captures run-level metrics (params, final scores, artifacts). It does *not* capture:

* Per-sample test predictions with uncertainty
* Layer-wise sparsity over training
* Component-level energy breakdowns (Eq. 25)
* Bidirectional iteration counts (Theorem 3.6 validation)

These fine-grained records are needed to generate the scientific report and validate theoretical claims. Storing them in a dedicated schema keeps MLflow clean while providing queryable research data.

---

## Applying Migrations

```bash
# Apply V001 (inside Postgres container)
docker compose exec postgres psql -U beam -f /sql/migrations/V001__beam_metrics_schema.sql

# Apply V002
docker compose exec postgres psql -U beam -f /sql/migrations/V002__add_indexes.sql
```

To auto-apply on startup, mount the `migrations/` folder into `/docker-entrypoint-initdb.d/` in `docker-compose.yml` (files run alphabetically).

---

## Running Queries

```bash
# Compare model variants
docker compose exec postgres psql -U beam -d beam_metrics \
  -v dataset="'nmnist'" -f /sql/queries/experiment_comparison.sql

# Reliability diagram data
docker compose exec postgres psql -U beam -d beam_metrics \
  -v experiment_id=1 -f /sql/queries/reliability_diagram.sql
```

---

## Design Principles

1. **Separation from MLflow** — custom schema never modifies MLflow tables
2. **Versioned migrations** — `V###__description.sql` naming convention (Flyway-style)
3. **Documented with COMMENTs** — every table explains its purpose and paper reference
4. **Indexed for query patterns** — partial indexes on common filters
5. **Referential integrity** — foreign keys with `ON DELETE CASCADE`
6. **Generated columns** — `is_correct` computed at storage, not query time
7. **Views for convenience** — `experiment_summary` denormalizes common joins
