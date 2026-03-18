use crate::error::{StoreError, StoreResult};
use crate::traits::Store;
use crate::types::{MemoryEntry, TimeRange};
use async_trait::async_trait;
use sqlx::SqlitePool;
use uuid::Uuid;

pub struct SqliteStore {
    pool: SqlitePool,
}

impl SqliteStore {
    pub async fn new(database_url: &str) -> StoreResult<Self> {
        let pool = SqlitePool::connect(database_url)
            .await
            .map_err(|e| StoreError::ConnectionFailed(e.to_string()))?;

        let store = Self { pool };
        store.run_migrations().await?;
        Ok(store)
    }

    async fn run_migrations(&self) -> StoreResult<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS memories (
                entry_id TEXT PRIMARY KEY,
                lossless_restatement TEXT NOT NULL,
                keywords TEXT,
                timestamp TEXT,
                location TEXT,
                persons TEXT,
                entities TEXT,
                topic TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::MigrationFailed(e.to_string()))?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)")
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::MigrationFailed(e.to_string()))?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_location ON memories(location)")
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::MigrationFailed(e.to_string()))?;

        Ok(())
    }
}

#[async_trait]
impl Store for SqliteStore {
    async fn save_entry(&self, entry: &MemoryEntry) -> StoreResult<()> {
        let keywords = serde_json::to_string(&entry.keywords).unwrap_or_else(|_| "[]".to_string());
        let persons = serde_json::to_string(&entry.persons).unwrap_or_else(|_| "[]".to_string());
        let entities = serde_json::to_string(&entry.entities).unwrap_or_else(|_| "[]".to_string());
        let timestamp = entry.timestamp.map(|t| t.to_rfc3339());
        let topic = entry.topic.clone();

        sqlx::query(
            r#"
            INSERT OR REPLACE INTO memories 
            (entry_id, lossless_restatement, keywords, timestamp, location, persons, entities, topic)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(entry.entry_id.to_string())
        .bind(&entry.lossless_restatement)
        .bind(&keywords)
        .bind(&timestamp)
        .bind(&entry.location)
        .bind(&persons)
        .bind(&entities)
        .bind(&topic)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::QueryFailed(e.to_string()))?;

        Ok(())
    }

    async fn get_entry(&self, entry_id: &Uuid) -> StoreResult<Option<MemoryEntry>> {
        let row: Option<MemoryEntryRow> =
            sqlx::query_as::<_, MemoryEntryRow>("SELECT * FROM memories WHERE entry_id = ?")
                .bind(entry_id.to_string())
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| StoreError::QueryFailed(e.to_string()))?;

        Ok(row.map(|r| r.into_entry()))
    }

    async fn delete_entry(&self, entry_id: &Uuid) -> StoreResult<bool> {
        let result = sqlx::query("DELETE FROM memories WHERE entry_id = ?")
            .bind(entry_id.to_string())
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::QueryFailed(e.to_string()))?;

        Ok(result.rows_affected() > 0)
    }

    async fn list_entries(&self, limit: Option<usize>) -> StoreResult<Vec<MemoryEntry>> {
        let query = match limit {
            Some(l) => format!("SELECT * FROM memories ORDER BY timestamp DESC LIMIT {}", l),
            None => "SELECT * FROM memories ORDER BY timestamp DESC".to_string(),
        };

        let rows = sqlx::query_as::<_, MemoryEntryRow>(&query)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| StoreError::QueryFailed(e.to_string()))?;

        Ok(rows.into_iter().map(|r| r.into_entry()).collect())
    }

    async fn search_by_metadata(
        &self,
        persons: Option<&[String]>,
        location: Option<&str>,
        entities: Option<&[String]>,
        _time_range: Option<&TimeRange>,
        limit: Option<usize>,
    ) -> StoreResult<Vec<MemoryEntry>> {
        let mut conditions = Vec::new();
        let mut params: Vec<String> = Vec::new();

        if let Some(loc) = location {
            conditions.push("location LIKE ?");
            params.push(format!("%{}%", loc));
        }

        if let Some(p) = persons {
            for person in p {
                conditions.push("persons LIKE ?");
                params.push(format!("%{}%", person));
            }
        }

        if let Some(e) = entities {
            for entity in e {
                conditions.push("entities LIKE ?");
                params.push(format!("%{}%", entity));
            }
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };

        let limit_clause = match limit {
            Some(l) => format!("LIMIT {}", l),
            None => String::new(),
        };

        let query = format!(
            "SELECT * FROM memories {} ORDER BY timestamp DESC {}",
            where_clause, limit_clause
        );

        let mut sql_query = sqlx::query_as::<_, MemoryEntryRow>(&query);
        for param in params {
            sql_query = sql_query.bind(param);
        }

        let rows = sql_query
            .fetch_all(&self.pool)
            .await
            .map_err(|e| StoreError::QueryFailed(e.to_string()))?;

        Ok(rows.into_iter().map(|r| r.into_entry()).collect())
    }
}

#[derive(sqlx::FromRow)]
struct MemoryEntryRow {
    entry_id: String,
    lossless_restatement: String,
    keywords: Option<String>,
    timestamp: Option<String>,
    location: Option<String>,
    persons: Option<String>,
    entities: Option<String>,
    topic: Option<String>,
}

impl MemoryEntryRow {
    fn into_entry(self) -> MemoryEntry {
        let entry_id = Uuid::parse_str(&self.entry_id).unwrap_or_else(|_| Uuid::new_v4());
        let keywords: Vec<String> = self
            .keywords
            .and_then(|k| serde_json::from_str(&k).ok())
            .unwrap_or_default();
        let persons: Vec<String> = self
            .persons
            .and_then(|p| serde_json::from_str(&p).ok())
            .unwrap_or_default();
        let entities: Vec<String> = self
            .entities
            .and_then(|e| serde_json::from_str(&e).ok())
            .unwrap_or_default();
        let timestamp = self
            .timestamp
            .and_then(|t| chrono::DateTime::parse_from_rfc3339(&t).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));

        MemoryEntry {
            entry_id,
            lossless_restatement: self.lossless_restatement,
            keywords,
            timestamp,
            location: self.location,
            persons,
            entities,
            topic: self.topic,
        }
    }
}
