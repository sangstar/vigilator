pub mod bindings;
pub mod db;
pub mod outputs;
mod server;

#[cfg(test)]
mod tests {
    use axum::body::{Body, to_bytes};
    use axum::http::{Request, StatusCode};
    use axum::Json;
    use serde_json::json;
    use crate::db::query_field;
    use crate::server::remove_leading_and_trailing_quotes;
    use crate::db::insert_output;
    use crate::outputs::ModelOutput;



    #[tokio::test]
    async fn test_get_top_token_logprob() {
        pyo3::prepare_freethreaded_python();

        let token_ids = vec![1, 2, 3];
        let logits = vec![0.1, 0.2, 0.3];
        let text = "foo bar".to_string();

        let output = ModelOutput::new(text.clone(), token_ids.clone(), logits.clone());
        insert_output(&output)
            .await
            .expect("Failed to insert output");

        let results = output.get_top_token_ids(2);
        assert_eq!(results.get(0).unwrap().1, 0.3);
        assert_eq!(results.get(0).unwrap().0, 3);

        assert_eq!(results.get(1).unwrap().1, 0.2);
        assert_eq!(results.get(1).unwrap().0, 2);
    }

    #[tokio::test]
    async fn test_json_endpoint() {
        pyo3::prepare_freethreaded_python();

        let token_ids = vec![1, 2, 3];
        let logits = vec![0.1, 0.2, 0.3];
        let text = "foo bar".to_string();

        let output = ModelOutput::new(text.clone(), token_ids.clone(), logits.clone());
        insert_output(&output)
            .await
            .expect("Failed to insert output");

        let payload = json!({ "text": text, "top_k": 2 });

        let request = Request::builder()
            .method("POST")
            .uri("/")
            .header("content-type", "application/json")
            .body(Body::from(payload.to_string()))
            .unwrap();

        let response = crate::server::handle_json(Json(payload)).await.unwrap();

        // Assert the response status and body
        assert_eq!(response.status(), StatusCode::OK);
        let parsed_body = to_bytes(response.into_body(), usize::MAX).await.expect("Failed to convert resp body to bytes");
        let json: serde_json::Value = serde_json::from_slice(&parsed_body).unwrap();

        dbg!(json.clone());

        let unprocessed_text = json.get("text").unwrap().to_string();
        assert_eq!(remove_leading_and_trailing_quotes(unprocessed_text).await, text);

        let top_k = json.get("top_k").unwrap().as_array().unwrap();

        dbg!(text);
        dbg!(top_k);
        let first_elem = &top_k[0];
        let as_tuple: (u32, f32) = serde_json::from_value(first_elem.clone()).expect("Failed to deserialize tuple");
        assert_eq!(as_tuple.0, 3);
        assert_eq!(as_tuple.1, 0.3);
        dbg!(as_tuple);


        let second_elem = &top_k[1];
        let as_tuple: (u32, f32) = serde_json::from_value(second_elem.clone()).expect("Failed to deserialize tuple");
        dbg!(as_tuple);
        assert_eq!(as_tuple.0, 2);
        assert_eq!(as_tuple.1, 0.2);
    }

    #[tokio::test]
    async fn test_send_output() {
        pyo3::prepare_freethreaded_python();

        let token_ids = vec![1, 2, 3];
        let logits = vec![0.1, 0.2, 0.3];
        let text = "foo bar".to_string();

        let output = ModelOutput::new(text.clone(), token_ids.clone(), logits.clone());
        insert_output(&output)
            .await
            .expect("Failed to insert output");

        let queried = query_field(output.text)
            .await
            .expect("Failed querying data.");

        assert_eq!(queried.text.value, text);
        assert_eq!(queried.token_ids.value, token_ids);
        assert_eq!(queried.logits.value, logits);
    }
}
