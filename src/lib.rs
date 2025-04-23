use crate::db::insert_output;
use crate::outputs::ModelOutput;

pub mod outputs;
pub mod bindings;
pub mod db;



#[cfg(test)]
mod tests {
    use super::*;
    use db::query_output_from_id;

    #[tokio::test]
    async fn test_send_output() {
        let token_ids = vec![1, 2, 3];
        let logits = vec![0.1, 0.2, 0.3];

        let output = ModelOutput::new(token_ids.clone(), logits.clone());
        insert_output(output).await.expect("Failed to insert output");

        let queried = query_output_from_id(1).await.expect("Failed querying data.");

        assert_eq!(queried.token_ids, token_ids);
        assert_eq!(queried.logits, logits);
    }
}
