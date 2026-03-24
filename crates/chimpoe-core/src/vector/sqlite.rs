use crate::error::{VectorError, VectorResult};
use crate::traits::VectorStore;
use crate::types::{MemoryEntry, StructuredSearchParams};
use async_trait::async_trait;
use rusqlite::{Connection, OptionalExtension};
use sqlite_vec::sqlite3_vec_init;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

const TABLE_NAME: &str = "memories";
const META_TABLE: &str = "memory_metadata";
const FTS_TABLE: &str = "memory_fts";
const CONFIG_TABLE: &str = "config";

static SQLITE_VEC_INIT: std::sync::Once = std::sync::Once::new();

pub struct SqliteVector {
    conn: Arc<Mutex<Connection>>,
    dimension: usize,
}

fn vec_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for &f in v {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    bytes
}

fn escape_fts5_query(s: &str) -> String {
    let escaped = s
        .replace('"', "\"\"")
        .replace(['*', '^'], "")
        .replace('\'', "''");
    format!("\"{}\"", escaped)
}

fn bytes_to_vec(bytes: &[u8], dimension: usize) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .take(dimension)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

impl SqliteVector {
    pub fn new(db_path: &str, dimension: usize) -> VectorResult<Self> {
        SQLITE_VEC_INIT.call_once(|| unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<
                *const (),
                unsafe extern "C" fn(
                    *mut rusqlite::ffi::sqlite3,
                    *mut *mut i8,
                    *const rusqlite::ffi::sqlite3_api_routines,
                ) -> i32,
            >(
                sqlite3_vec_init as *const ()
            )));
        });

        let conn = Connection::open(db_path)
            .map_err(|e| VectorError::ConnectionFailed(format!("Failed to open database: {e}")))?;

        conn.execute_batch(
            r"
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA foreign_keys=ON;
            PRAGMA busy_timeout=5000;
            ",
        )
        .map_err(|e| VectorError::ConnectionFailed(format!("Failed to set PRAGMA: {e}")))?;

        conn.execute_batch(&format!(
            r#"
            CREATE TABLE IF NOT EXISTS {CONFIG_TABLE} (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            "#
        ))
        .map_err(|e| VectorError::IndexCreationFailed(e.to_string()))?;

        let stored_dimension: Option<usize> = conn
            .query_row(
                &format!("SELECT value FROM {CONFIG_TABLE} WHERE key = 'dimension'"),
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(|e| VectorError::ConnectionFailed(format!("Failed to read config: {e}")))?
            .and_then(|s| s.parse().ok());

        let dimension = match stored_dimension {
            Some(db_dim) => {
                if db_dim != dimension {
                    return Err(VectorError::DimensionMismatchOnOpen {
                        db_dimension: db_dim,
                        requested_dimension: dimension,
                    });
                }
                db_dim
            }
            None => {
                conn.execute(
                    &format!("INSERT INTO {CONFIG_TABLE} (key, value) VALUES ('dimension', ?1)"),
                    [dimension.to_string()],
                )
                .map_err(|e| {
                    VectorError::ConnectionFailed(format!("Failed to store dimension: {e}"))
                })?;
                dimension
            }
        };

        conn.execute_batch(&format!(
            r#"
            CREATE TABLE IF NOT EXISTS {META_TABLE} (
                id TEXT PRIMARY KEY,
                lossless_restatement TEXT NOT NULL,
                keywords TEXT NOT NULL,
                persons TEXT NOT NULL,
                entities TEXT NOT NULL,
                location TEXT,
                topic TEXT,
                timestamp TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_timestamp ON {META_TABLE}(timestamp);
            CREATE INDEX IF NOT EXISTS idx_location ON {META_TABLE}(location);
            CREATE INDEX IF NOT EXISTS idx_topic ON {META_TABLE}(topic);

            CREATE VIRTUAL TABLE IF NOT EXISTS {TABLE_NAME} USING vec0(
                embedding float[{dimension}]
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE} USING fts5(
                lossless_restatement,
                keywords,
                content={META_TABLE},
                content_rowid=rowid,
                tokenize="porter unicode61"
            );

            CREATE TRIGGER IF NOT EXISTS memory_fts_insert AFTER INSERT ON {META_TABLE} BEGIN
                INSERT INTO {FTS_TABLE}(rowid, lossless_restatement, keywords)
                VALUES (new.rowid, new.lossless_restatement, new.keywords);
            END;

            CREATE TRIGGER IF NOT EXISTS memory_fts_delete AFTER DELETE ON {META_TABLE} BEGIN
                INSERT INTO {FTS_TABLE}({FTS_TABLE}, rowid, lossless_restatement, keywords)
                VALUES('delete', old.rowid, old.lossless_restatement, old.keywords);
            END;

            CREATE TRIGGER IF NOT EXISTS memory_fts_update AFTER UPDATE ON {META_TABLE} BEGIN
                INSERT INTO {FTS_TABLE}({FTS_TABLE}, rowid, lossless_restatement, keywords)
                VALUES('delete', old.rowid, old.lossless_restatement, old.keywords);
                INSERT INTO {FTS_TABLE}(rowid, lossless_restatement, keywords)
                VALUES (new.rowid, new.lossless_restatement, new.keywords);
            END;
            "#
        ))
        .map_err(|e| VectorError::IndexCreationFailed(e.to_string()))?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            dimension,
        })
    }

    pub fn in_memory(dimension: usize) -> VectorResult<Self> {
        Self::new(":memory:", dimension)
    }

    fn parse_entry_from_row(row: &rusqlite::Row) -> Result<MemoryEntry, rusqlite::Error> {
        let id_str: String = row.get("id")?;
        let entry_id = Uuid::parse_str(&id_str).map_err(|_| rusqlite::Error::InvalidQuery)?;

        let keywords_str: String = row.get("keywords")?;
        let persons_str: String = row.get("persons")?;
        let entities_str: String = row.get("entities")?;
        let timestamp_str: Option<String> = row.get("timestamp")?;

        Ok(MemoryEntry {
            entry_id,
            lossless_restatement: row.get("lossless_restatement")?,
            keywords: serde_json::from_str(&keywords_str).unwrap_or_default(),
            persons: serde_json::from_str(&persons_str).unwrap_or_default(),
            entities: serde_json::from_str(&entities_str).unwrap_or_default(),
            location: row.get("location")?,
            topic: row.get("topic")?,
            timestamp: timestamp_str.and_then(|s| {
                chrono::DateTime::parse_from_rfc3339(&s)
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .ok()
            }),
        })
    }
}

#[async_trait]
impl VectorStore for SqliteVector {
    async fn add_entries(&self, entries: &[MemoryEntry], vectors: &[Vec<f32>]) -> VectorResult<()> {
        if entries.len() != vectors.len() {
            return Err(VectorError::InsertionFailed(
                "entries and vectors length mismatch".to_string(),
            ));
        }

        let conn = self.conn.lock().await;

        conn.execute("BEGIN", [])
            .map_err(|e| VectorError::InsertionFailed(e.to_string()))?;

        let result = (|| {
            let insert_meta_sql = format!(
                "INSERT INTO {META_TABLE} (id, lossless_restatement, keywords, persons, entities, location, topic, timestamp)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)"
            );
            let mut insert_meta_stmt = conn
                .prepare_cached(&insert_meta_sql)
                .map_err(|e| VectorError::InsertionFailed(e.to_string()))?;

            let insert_vec_sql =
                format!("INSERT INTO {TABLE_NAME} (rowid, embedding) VALUES (?1, ?2)");
            let mut insert_vec_stmt = conn
                .prepare_cached(&insert_vec_sql)
                .map_err(|e| VectorError::InsertionFailed(e.to_string()))?;

            for (entry, vector) in entries.iter().zip(vectors.iter()) {
                if vector.len() != self.dimension {
                    return Err(VectorError::DimensionMismatch {
                        expected: self.dimension,
                        actual: vector.len(),
                    });
                }

                let timestamp_str = entry.timestamp.as_ref().map(chrono::DateTime::to_rfc3339);

                insert_meta_stmt
                    .execute(rusqlite::params![
                        entry.entry_id.to_string(),
                        entry.lossless_restatement,
                        serde_json::to_string(&entry.keywords).unwrap_or_else(|_| "[]".to_string()),
                        serde_json::to_string(&entry.persons).unwrap_or_else(|_| "[]".to_string()),
                        serde_json::to_string(&entry.entities).unwrap_or_else(|_| "[]".to_string()),
                        entry.location.as_ref(),
                        entry.topic.as_ref(),
                        timestamp_str,
                    ])
                    .map_err(|e| VectorError::InsertionFailed(e.to_string()))?;

                let rowid = conn.last_insert_rowid();

                insert_vec_stmt
                    .execute(rusqlite::params![rowid, vec_to_bytes(vector)])
                    .map_err(|e| VectorError::InsertionFailed(e.to_string()))?;
            }
            Ok(())
        })();

        match result {
            Ok(()) => {
                conn.execute("COMMIT", [])
                    .map_err(|e| VectorError::InsertionFailed(e.to_string()))?;
                Ok(())
            }
            Err(e) => {
                let _ = conn.execute("ROLLBACK", []);
                Err(e)
            }
        }
    }

    async fn semantic_search(
        &self,
        query_vector: &[f32],
        top_k: usize,
    ) -> VectorResult<Vec<MemoryEntry>> {
        if query_vector.len() != self.dimension {
            return Err(VectorError::DimensionMismatch {
                expected: self.dimension,
                actual: query_vector.len(),
            });
        }

        let conn = self.conn.lock().await;

        let mut stmt = conn
            .prepare_cached(&format!(
                r"
                SELECT m.id, m.lossless_restatement, m.keywords, m.persons, m.entities,
                       m.location, m.topic, m.timestamp
                FROM {TABLE_NAME} v
                JOIN {META_TABLE} m ON v.rowid = m.rowid
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
                "
            ))
            .map_err(|e| VectorError::SearchFailed(e.to_string()))?;

        let rows: Vec<std::result::Result<MemoryEntry, rusqlite::Error>> = stmt
            .query_map(
                rusqlite::params![vec_to_bytes(query_vector), top_k as i64],
                Self::parse_entry_from_row,
            )
            .map_err(|e| VectorError::SearchFailed(e.to_string()))?
            .collect();

        rows.into_iter()
            .map(|r| r.map_err(|e| VectorError::SearchFailed(e.to_string())))
            .collect()
    }

    async fn keyword_search(
        &self,
        keywords: &[String],
        top_k: usize,
    ) -> VectorResult<Vec<MemoryEntry>> {
        if keywords.is_empty() {
            return Ok(Vec::new());
        }

        let conn = self.conn.lock().await;

        let fts_query: String = keywords
            .iter()
            .map(|kw| escape_fts5_query(kw))
            .collect::<Vec<_>>()
            .join(" OR ");

        let mut stmt = conn
            .prepare_cached(&format!(
                r"
                SELECT m.id, m.lossless_restatement, m.keywords, m.persons, m.entities,
                       m.location, m.topic, m.timestamp
                FROM {FTS_TABLE} fts
                JOIN {META_TABLE} m ON fts.rowid = m.rowid
                WHERE {FTS_TABLE} MATCH ?
                ORDER BY bm25(memory_fts)
                LIMIT ?
                "
            ))
            .map_err(|e| VectorError::SearchFailed(e.to_string()))?;

        let rows: Vec<std::result::Result<MemoryEntry, rusqlite::Error>> = stmt
            .query_map(rusqlite::params![fts_query, top_k as i64], |row| {
                Self::parse_entry_from_row(row)
            })
            .map_err(|e| VectorError::SearchFailed(e.to_string()))?
            .collect();

        rows.into_iter()
            .map(|r| r.map_err(|e| VectorError::SearchFailed(e.to_string())))
            .collect()
    }

    async fn structured_search(
        &self,
        params: &StructuredSearchParams,
        top_k: usize,
    ) -> VectorResult<Vec<MemoryEntry>> {
        let has_persons = params.persons.as_ref().is_some_and(|p| !p.is_empty());
        let has_entities = params.entities.as_ref().is_some_and(|e| !e.is_empty());
        let has_location = params.location.is_some();
        let has_time = params.timestamp_range.is_some();

        if !has_persons && !has_entities && !has_location && !has_time {
            return Ok(Vec::new());
        }

        let conn = self.conn.lock().await;
        let mut conditions: Vec<String> = Vec::new();
        let mut bind_params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

        if let Some(ref persons) = params.persons
            && !persons.is_empty()
        {
            let placeholders: Vec<String> = persons.iter().map(|_| "?".to_string()).collect();
            conditions.push(format!(
                "EXISTS (SELECT 1 FROM json_each(persons) WHERE value IN ({}))",
                placeholders.join(", ")
            ));
            for person in persons {
                bind_params.push(Box::new(person.clone()));
            }
        }

        if let Some(ref location) = params.location {
            conditions.push("location = ?".to_string());
            bind_params.push(Box::new(location.clone()));
        }

        if let Some(ref entities) = params.entities
            && !entities.is_empty()
        {
            let placeholders: Vec<String> = entities.iter().map(|_| "?".to_string()).collect();
            conditions.push(format!(
                "EXISTS (SELECT 1 FROM json_each(entities) WHERE value IN ({}))",
                placeholders.join(", ")
            ));
            for entity in entities {
                bind_params.push(Box::new(entity.clone()));
            }
        }

        if let Some(ref time_range) = params.timestamp_range {
            conditions.push("timestamp >= ? AND timestamp <= ?".to_string());
            bind_params.push(Box::new(time_range.start.to_rfc3339()));
            bind_params.push(Box::new(time_range.end.to_rfc3339()));
        }

        let where_clause = conditions.join(" AND ");

        let query = format!(
            "SELECT id, lossless_restatement, keywords, persons, entities, location, topic, timestamp FROM {META_TABLE} WHERE {where_clause} ORDER BY timestamp DESC LIMIT ?"
        );

        bind_params.push(Box::new(top_k as i64));

        let params_refs: Vec<&dyn rusqlite::ToSql> =
            bind_params.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn
            .prepare(&query)
            .map_err(|e| VectorError::SearchFailed(e.to_string()))?;

        let rows: Vec<std::result::Result<MemoryEntry, rusqlite::Error>> = stmt
            .query_map(params_refs.as_slice(), Self::parse_entry_from_row)
            .map_err(|e| VectorError::SearchFailed(e.to_string()))?
            .collect();

        rows.into_iter()
            .map(|r| r.map_err(|e| VectorError::SearchFailed(e.to_string())))
            .collect()
    }

    async fn delete_entry(&self, entry_id: &Uuid) -> VectorResult<bool> {
        let conn = self.conn.lock().await;

        let rowid: Option<i64> = conn
            .query_row(
                &format!("SELECT rowid FROM {META_TABLE} WHERE id = ?1"),
                rusqlite::params![entry_id.to_string()],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| VectorError::DeletionFailed(e.to_string()))?;

        let Some(rowid) = rowid else {
            return Ok(false);
        };

        conn.execute("BEGIN", [])
            .map_err(|e| VectorError::DeletionFailed(e.to_string()))?;

        let result = (|| {
            let mut delete_meta_stmt = conn
                .prepare_cached(&format!("DELETE FROM {META_TABLE} WHERE id = ?1"))
                .map_err(|e| VectorError::DeletionFailed(e.to_string()))?;
            delete_meta_stmt
                .execute(rusqlite::params![entry_id.to_string()])
                .map_err(|e| VectorError::DeletionFailed(e.to_string()))?;

            let mut delete_vec_stmt = conn
                .prepare_cached(&format!("DELETE FROM {TABLE_NAME} WHERE rowid = ?1"))
                .map_err(|e| VectorError::DeletionFailed(e.to_string()))?;
            delete_vec_stmt
                .execute(rusqlite::params![rowid])
                .map_err(|e| VectorError::DeletionFailed(e.to_string()))?;

            Ok(())
        })();

        match result {
            Ok(()) => {
                conn.execute("COMMIT", [])
                    .map_err(|e| VectorError::DeletionFailed(e.to_string()))?;
                Ok(true)
            }
            Err(e) => {
                let _ = conn.execute("ROLLBACK", []);
                Err(e)
            }
        }
    }

    async fn count(&self) -> VectorResult<usize> {
        let conn = self.conn.lock().await;

        let count: i64 = conn
            .query_row(&format!("SELECT COUNT(*) FROM {META_TABLE}"), [], |row| {
                row.get(0)
            })
            .map_err(|e| VectorError::SearchFailed(e.to_string()))?;

        Ok(count as usize)
    }

    async fn get_all_entries(&self) -> VectorResult<Vec<MemoryEntry>> {
        let conn = self.conn.lock().await;

        let mut stmt = conn
            .prepare_cached(&format!(
                "SELECT id, lossless_restatement, keywords, persons, entities, location, topic, timestamp FROM {META_TABLE}"
            ))
            .map_err(|e| VectorError::SearchFailed(e.to_string()))?;

        let rows: Vec<std::result::Result<MemoryEntry, rusqlite::Error>> = stmt
            .query_map([], Self::parse_entry_from_row)
            .map_err(|e| VectorError::SearchFailed(e.to_string()))?
            .collect();

        rows.into_iter()
            .map(|r| r.map_err(|e| VectorError::SearchFailed(e.to_string())))
            .collect()
    }

    async fn get_all_entries_with_vectors(&self) -> VectorResult<Vec<(MemoryEntry, Vec<f32>)>> {
        let conn = self.conn.lock().await;

        let mut stmt = conn
            .prepare_cached(&format!(
                r"
                SELECT m.id, m.lossless_restatement, m.keywords, m.persons, m.entities,
                       m.location, m.topic, m.timestamp, v.embedding AS embedding
                FROM {META_TABLE} m
                JOIN {TABLE_NAME} v ON m.rowid = v.rowid
                "
            ))
            .map_err(|e| VectorError::SearchFailed(e.to_string()))?;

        let rows: Vec<std::result::Result<(MemoryEntry, Vec<f32>), rusqlite::Error>> = stmt
            .query_map([], |row| {
                let entry = Self::parse_entry_from_row(row)?;
                let embedding_bytes: Vec<u8> = row.get("embedding")?;
                let vector = bytes_to_vec(&embedding_bytes, self.dimension);
                Ok((entry, vector))
            })
            .map_err(|e| VectorError::SearchFailed(e.to_string()))?
            .collect();

        rows.into_iter()
            .map(|r| r.map_err(|e| VectorError::SearchFailed(e.to_string())))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::VectorStore;

    fn make_entry(text: &str) -> MemoryEntry {
        MemoryEntry::new(text.to_string()).with_keywords(vec!["test".to_string()])
    }

    fn make_vector(dim: usize) -> Vec<f32> {
        vec![0.1; dim]
    }

    #[tokio::test]
    async fn test_in_memory_creation() {
        let store = SqliteVector::in_memory(128).unwrap();
        assert_eq!(store.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_add_single_entry() {
        let store = SqliteVector::in_memory(64).unwrap();
        let entry = make_entry("Test memory");
        let vector = make_vector(64);

        store.add_entries(&[entry], &[vector]).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_add_multiple_entries() {
        let store = SqliteVector::in_memory(64).unwrap();
        let entries = vec![
            make_entry("Memory 1"),
            make_entry("Memory 2"),
            make_entry("Memory 3"),
        ];
        let vectors = vec![make_vector(64), make_vector(64), make_vector(64)];

        store.add_entries(&entries, &vectors).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 3);
    }

    #[tokio::test]
    async fn test_add_entries_mismatch_length() {
        let store = SqliteVector::in_memory(64).unwrap();
        let entries = vec![make_entry("Memory 1")];
        let vectors = vec![make_vector(64), make_vector(64)];

        let result = store.add_entries(&entries, &vectors).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_add_entry_dimension_mismatch() {
        let store = SqliteVector::in_memory(64).unwrap();
        let entry = make_entry("Test");
        let wrong_dim_vector = vec![0.1; 32];

        let result = store.add_entries(&[entry], &[wrong_dim_vector]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_semantic_search_empty() {
        let store = SqliteVector::in_memory(64).unwrap();
        let query = make_vector(64);

        let results = store.semantic_search(&query, 5).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_semantic_search_returns_results() {
        let store = SqliteVector::in_memory(64).unwrap();
        let entry = make_entry("Semantic test memory");
        let vector = make_vector(64);

        store.add_entries(&[entry], &[vector]).await.unwrap();

        let query = make_vector(64);
        let results = store.semantic_search(&query, 5).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].lossless_restatement, "Semantic test memory");
    }

    #[tokio::test]
    async fn test_semantic_search_respects_top_k() {
        let store = SqliteVector::in_memory(64).unwrap();
        let entries: Vec<MemoryEntry> = (0..10)
            .map(|i| make_entry(&format!("Memory {i}")))
            .collect();
        let vectors: Vec<Vec<f32>> = (0..10).map(|_| make_vector(64)).collect();

        store.add_entries(&entries, &vectors).await.unwrap();

        let query = make_vector(64);
        let results = store.semantic_search(&query, 3).await.unwrap();
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn test_keyword_search_empty_store() {
        let store = SqliteVector::in_memory(64).unwrap();

        let results = store
            .keyword_search(&["test".to_string()], 5)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_keyword_search_empty_keywords() {
        let store = SqliteVector::in_memory(64).unwrap();
        let entry = make_entry("Test memory");
        let vector = make_vector(64);

        store.add_entries(&[entry], &[vector]).await.unwrap();

        let results = store.keyword_search(&[], 5).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_keyword_search_finds_match() {
        let store = SqliteVector::in_memory(64).unwrap();
        let entry = MemoryEntry::new("User loves programming in Rust".to_string())
            .with_keywords(vec!["programming".to_string(), "rust".to_string()]);
        let vector = make_vector(64);

        store.add_entries(&[entry], &[vector]).await.unwrap();

        let results = store
            .keyword_search(&["programming".to_string()], 5)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_keyword_search_no_match() {
        let store = SqliteVector::in_memory(64).unwrap();
        let entry = make_entry("User likes pizza");
        let vector = make_vector(64);

        store.add_entries(&[entry], &[vector]).await.unwrap();

        let results = store
            .keyword_search(&["unrelated".to_string()], 5)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_structured_search_empty_params() {
        let store = SqliteVector::in_memory(64).unwrap();
        let entry = MemoryEntry::new("Test".to_string()).with_persons(vec!["Alice".to_string()]);
        let vector = make_vector(64);

        store.add_entries(&[entry], &[vector]).await.unwrap();

        let params = StructuredSearchParams::default();
        let results = store.structured_search(&params, 5).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_structured_search_by_person() {
        let store = SqliteVector::in_memory(64).unwrap();
        let entry = MemoryEntry::new("Meeting with Alice".to_string())
            .with_persons(vec!["Alice".to_string(), "Bob".to_string()]);
        let vector = make_vector(64);

        store.add_entries(&[entry], &[vector]).await.unwrap();

        let params = StructuredSearchParams {
            persons: Some(vec!["Alice".to_string()]),
            ..Default::default()
        };
        let results = store.structured_search(&params, 5).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_structured_search_by_location() {
        let store = SqliteVector::in_memory(64).unwrap();
        let entry =
            MemoryEntry::new("Meeting at office".to_string()).with_location("Jakarta".to_string());
        let vector = make_vector(64);

        store.add_entries(&[entry], &[vector]).await.unwrap();

        let params = StructuredSearchParams {
            location: Some("Jakarta".to_string()),
            ..Default::default()
        };
        let results = store.structured_search(&params, 5).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_structured_search_by_entity() {
        let store = SqliteVector::in_memory(64).unwrap();
        let entry = MemoryEntry::new("Working on ProjectX".to_string())
            .with_entities(vec!["ProjectX".to_string()]);
        let vector = make_vector(64);

        store.add_entries(&[entry], &[vector]).await.unwrap();

        let params = StructuredSearchParams {
            entities: Some(vec!["ProjectX".to_string()]),
            ..Default::default()
        };
        let results = store.structured_search(&params, 5).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_delete_entry() {
        let store = SqliteVector::in_memory(64).unwrap();
        let entry = make_entry("To be deleted");
        let entry_id = entry.entry_id;
        let vector = make_vector(64);

        store.add_entries(&[entry], &[vector]).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 1);

        let deleted = store.delete_entry(&entry_id).await.unwrap();
        assert!(deleted);
        assert_eq!(store.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_delete_nonexistent_entry() {
        let store = SqliteVector::in_memory(64).unwrap();
        let fake_id = Uuid::new_v4();

        let deleted = store.delete_entry(&fake_id).await.unwrap();
        assert!(!deleted);
    }

    #[tokio::test]
    async fn test_get_all_entries() {
        let store = SqliteVector::in_memory(64).unwrap();
        let entries = vec![make_entry("A"), make_entry("B"), make_entry("C")];
        let vectors = vec![make_vector(64), make_vector(64), make_vector(64)];

        store.add_entries(&entries, &vectors).await.unwrap();

        let all = store.get_all_entries().await.unwrap();
        assert_eq!(all.len(), 3);
    }

    #[tokio::test]
    async fn test_get_all_entries_empty() {
        let store = SqliteVector::in_memory(64).unwrap();

        let all = store.get_all_entries().await.unwrap();
        assert!(all.is_empty());
    }

    #[tokio::test]
    async fn test_entry_with_all_fields() {
        let store = SqliteVector::in_memory(64).unwrap();
        let entry = MemoryEntry::new("Full entry test".to_string())
            .with_keywords(vec!["keyword1".to_string(), "keyword2".to_string()])
            .with_persons(vec!["Alice".to_string()])
            .with_entities(vec!["ProjectX".to_string()])
            .with_location("Jakarta".to_string())
            .with_topic("Testing".to_string());
        let vector = make_vector(64);

        store
            .add_entries(std::slice::from_ref(&entry), &[vector])
            .await
            .unwrap();

        let all = store.get_all_entries().await.unwrap();
        assert_eq!(all.len(), 1);

        let retrieved = &all[0];
        assert_eq!(retrieved.lossless_restatement, "Full entry test");
        assert!(retrieved.keywords.contains(&"keyword1".to_string()));
        assert!(retrieved.persons.contains(&"Alice".to_string()));
        assert!(retrieved.entities.contains(&"ProjectX".to_string()));
        assert_eq!(retrieved.location, Some("Jakarta".to_string()));
        assert_eq!(retrieved.topic, Some("Testing".to_string()));
    }
}
