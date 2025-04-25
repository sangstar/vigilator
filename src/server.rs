use axum::{
    extract::{Json, Path},
    http::{header, StatusCode},
    body::Body,
    routing::get,
    response::Response,
    Router,
};
use axum::response::IntoResponse;
use serde_json::{Value, json};
use crate::db::query_field;
use crate::outputs::{Field, FieldName, ModelOutput};


#[tokio::main]
async fn main() {
    let app = Router::new().route("/", get(handle_json));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

pub async fn remove_leading_and_trailing_quotes(text: String) -> String {
    let replaced = text.replace("\"", "");
    replaced
}

pub async fn handle_json(Json(payload): Json<serde_json::Value>) -> Result<Response, StatusCode> {

    if let Some(text) = payload.get("text") {
        let parsed_text = remove_leading_and_trailing_quotes(text.to_string()).await;
        let output: ModelOutput = query_field(Field { field_name: FieldName::Text, value: parsed_text}).await.unwrap();

        if let Some(top_k) = payload.get("top_k") {
            let top_tokens = output.get_top_token_ids(top_k.as_u64().unwrap());
            Ok(Json(json!({ "text": text, "logits": output.logits, "token_ids": output.token_ids , "top_k": top_tokens })).into_response())
        } else {
            Ok(Json(json!(
                { "text": text, "logits": output.logits, "token_ids": output.token_ids }
            )).into_response())
        }
    } else {
        Err(StatusCode::BAD_REQUEST)
    }
}
