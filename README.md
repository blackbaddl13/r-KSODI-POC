# KSODI POC based on Langchain Template 


Bulk export in Apache Parquet

Langsmith Destination ID

addd518a-2d68-4642-b34b-c25b68186294

```shell
curl --request POST \
  --url 'https://eu.api.smith.langchain.com/api/v1/bulk-exports/destinations' \
  --header 'Content-Type: application/json' \
  --header 'X-API-Key: YOUR_API_KEY' \
  --header 'X-Tenant-Id: 5e3b4eca-620c-407c-af0b-193c57985d4b' \
  --data '{
    "destination_type": "s3",
    "display_name": "IONOS Berlin",
    "config": {
      "bucket_name": "ksodi-poc-255233211",
      "prefix": "langsmith_KSODI",
      "endpoint_url": "https://s3-eu-central-2.ionoscloud.com"
    },
    "credentials": {
      "access_key_id": "YOUR_S3_ACCESS_KEY_ID",
      "secret_access_key": "YOUR_S3_SECRET_ACCESS_KEY"
    }
  }'
```


```shell
curl --request POST \
  --url 'https://eu.api.smith.langchain.com/api/v1/bulk-exports' \
  --header 'Content-Type: application/json' \
  --header 'X-API-Key: YOUR_API_KEY' \
  --header 'X-Tenant-Id: 5e3b4eca-620c-407c-af0b-193c57985d4b' \
  --data '{
    "bulk_export_destination_id": "your_destination_id",
    "session_id": "project_uuid",
    "start_time": "2024-01-01T00:00:00Z",
    "interval_hours": 24
  }'
```