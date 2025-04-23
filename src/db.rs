use once_cell::sync::{Lazy, OnceCell};
use pyo3::PyErr;
use sqlx;
use sqlx::SqlitePool;
use tokio;
use tokio::runtime::Runtime;
use crate::outputs::ModelOutput;

pub static TOKIO_RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    Runtime::new().expect("Failed to create tokio runtime")
});

static SQLITE_POOL: OnceCell<SqlitePool> = OnceCell::new();


async fn debug_table_contents(pool: &SqlitePool) {
    let rows = sqlx::query_as::<_, (i32, Vec<u8>)>(
        r#"
        SELECT id, output
        FROM model_outputs
        "#
    )
        .fetch_all(pool)
        .await
        .expect("Failed to fetch table contents");

    for (id, output) in rows {
        println!("id: {}, output: {:?}", id, output);
    }
}

async fn init_pool() {
    let pool = SqlitePool::connect("sqlite::memory:").await.expect("Failed to connect");
    if SQLITE_POOL.set(pool).is_err() {
        eprintln!("Pool already set.");
        debug_table_contents(SQLITE_POOL.get().unwrap()).await;
    }
}

async fn maybe_create_table(pool: &SqlitePool) -> Result<(), sqlx::Error> {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS model_outputs (
            id INTEGER PRIMARY KEY,
            output TEXT NOT NULL
        )
        "#,
    )
        .execute(pool)
        .await?;
    Ok(())
}

async fn maybe_insert_output(pool: &SqlitePool, output: ModelOutput) -> Result<(), sqlx::Error> {
    let serialized = output.serialize().map_err(|e| {
        sqlx::Error::Decode(sqlx::error::BoxDynError::from(e))
    })?;
    sqlx::query(
        r#"
        INSERT INTO model_outputs (output)
        VALUES (?)
        "#,
    )
        .bind(serialized)
        .execute(pool)
        .await?;
    Ok(())
}

pub async fn insert_output(output: ModelOutput) -> Result<(), sqlx::Error> {
    init_pool().await;
    let pool = &SQLITE_POOL.get().unwrap();
    maybe_create_table(pool).await?;
    maybe_insert_output(pool, output).await?;
    Ok(())
}


pub async fn query_output_from_id(id: i32) -> Result<ModelOutput, PyErr> {
    init_pool().await;
    let pool = &SQLITE_POOL.get().unwrap();
    dbg!("Trying to query from {}", id);
    let row: (Vec<u8>,) = sqlx::query_as::<_, (Vec<u8>,)>(
        r#"
        SELECT output
        FROM model_outputs
        WHERE id = (?)
        "#,
    )
        .bind(id)
        .fetch_one(*pool)
        .await
        .map_err(|_e| { PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to retrieve query.") })?;

    ModelOutput::deserialize(&row.0).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to deserialize: {}", e))
    })
}


#[tokio::test]
async fn test_send_output() {
    let pool = SqlitePool::connect("sqlite::memory:").await.expect("Failed to connect");
    SQLITE_POOL.set(pool).expect("Pool already initialized");

    let token_ids = vec![1, 2, 3];
    let logits = vec![0.1, 0.2, 0.3];

    let output = ModelOutput::new(token_ids.clone(), logits.clone());
    insert_output(output).await.expect("Failed to insert output");

    let row: (Vec<u8>,) = sqlx::query_as(
        r#"
        SELECT output
        FROM model_outputs
        WHERE id = 1
        "#,
    )
        .fetch_one(&*SQLITE_POOL.get().unwrap())
        .await
        .expect("Failed to fetch output");

    let deserialized: ModelOutput = ModelOutput::deserialize(&row.0).expect("Failed to deserialize");
    assert_eq!(deserialized.token_ids, token_ids);
    assert_eq!(deserialized.logits, logits);
}