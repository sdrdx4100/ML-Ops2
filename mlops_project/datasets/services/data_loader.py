"""
Data loader service for managing datasets.
"""
import os
import duckdb
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from django.conf import settings
from django.core.files.uploadedfile import UploadedFile

from datasets.models import Dataset
from shared.utils import get_logger
from shared.utils.exceptions import DataError

logger = get_logger(__name__)


class DataLoader:
    """
    Service for loading and managing datasets.
    
    Supports CSV and Parquet files, with DuckDB for efficient querying.
    """
    
    def __init__(self, datasets_dir: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            datasets_dir: Directory to store dataset files
        """
        self.datasets_dir = datasets_dir or os.path.join(
            settings.MEDIA_ROOT, 'datasets'
        )
        os.makedirs(self.datasets_dir, exist_ok=True)
        
        # Initialize DuckDB connection
        self.db_path = os.path.join(self.datasets_dir, 'datasets.duckdb')
        self.conn = duckdb.connect(self.db_path)
    
    def _compute_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Compute statistics for each column."""
        stats = {}
        
        for col in df.columns:
            col_stats = {
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isnull().sum()),
                'unique_count': int(df[col].nunique()),
            }
            
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                col_stats.update({
                    'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                    'std': float(df[col].std()) if not df[col].isnull().all() else None,
                    'min': float(df[col].min()) if not df[col].isnull().all() else None,
                    'max': float(df[col].max()) if not df[col].isnull().all() else None,
                    'median': float(df[col].median()) if not df[col].isnull().all() else None,
                })
            
            stats[col] = col_stats
        
        return stats
    
    def upload(
        self,
        file: UploadedFile,
        name: str,
        file_type: Optional[str] = None
    ) -> Dataset:
        """
        Upload a dataset file.
        
        Args:
            file: Uploaded file object
            name: Dataset name
            file_type: File type (csv or parquet), auto-detected if not provided
        
        Returns:
            Created Dataset instance
        """
        # Detect file type
        if file_type is None:
            filename = file.name.lower()
            if filename.endswith('.parquet'):
                file_type = 'parquet'
            else:
                file_type = 'csv'
        
        # Save file
        safe_name = name.replace(' ', '_').lower()
        ext = 'parquet' if file_type == 'parquet' else 'csv'
        filename = f"{safe_name}.{ext}"
        filepath = os.path.join(self.datasets_dir, filename)
        
        with open(filepath, 'wb') as f:
            for chunk in file.chunks():
                f.write(chunk)
        
        # Read data
        if file_type == 'parquet':
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath)
        
        # Compute metadata
        row_count = len(df)
        column_names = df.columns.tolist()
        statistics = self._compute_statistics(df)
        
        # Register in DuckDB
        table_name = safe_name
        self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
        
        # Create or update dataset record
        dataset, created = Dataset.objects.update_or_create(
            name=name,
            defaults={
                'file_path': filepath,
                'file_type': file_type,
                'row_count': row_count,
                'column_names': column_names,
                'statistics': statistics,
            }
        )
        
        logger.info(f"Uploaded dataset '{name}' with {row_count} rows")
        return dataset
    
    def load(
        self,
        name: str,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load a dataset.
        
        Args:
            name: Dataset name
            columns: Optional list of columns to load
            limit: Optional row limit
        
        Returns:
            DataFrame with the data
        """
        try:
            dataset = Dataset.objects.get(name=name)
        except Dataset.DoesNotExist:
            raise DataError(f"Dataset '{name}' not found")
        
        # Try DuckDB first
        safe_name = name.replace(' ', '_').lower()
        try:
            cols = ', '.join(columns) if columns else '*'
            query = f"SELECT {cols} FROM {safe_name}"
            if limit:
                query += f" LIMIT {limit}"
            
            df = self.conn.execute(query).fetchdf()
            return df
        except Exception:
            # Fall back to file
            pass
        
        # Load from file
        if dataset.file_type == 'parquet':
            df = pd.read_parquet(dataset.file_path)
        else:
            df = pd.read_csv(dataset.file_path)
        
        if columns:
            df = df[columns]
        if limit:
            df = df.head(limit)
        
        return df
    
    def get_preview(self, name: str, max_rows: int = 100) -> pd.DataFrame:
        """
        Get a preview of a dataset.
        
        Args:
            name: Dataset name
            max_rows: Maximum number of rows to return
        
        Returns:
            DataFrame with preview data
        """
        return self.load(name, limit=max_rows)
    
    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute a SQL query against the datasets.
        
        Args:
            sql: SQL query string
        
        Returns:
            DataFrame with query results
        """
        try:
            return self.conn.execute(sql).fetchdf()
        except Exception as e:
            raise DataError(f"Query execution failed: {str(e)}")
    
    def list_datasets(self) -> List[Dataset]:
        """
        List all available datasets.
        
        Returns:
            List of Dataset instances
        """
        return list(Dataset.objects.all())
    
    def delete(self, name: str, delete_files: bool = True) -> None:
        """
        Delete a dataset.
        
        Args:
            name: Dataset name
            delete_files: Whether to delete the data file
        """
        try:
            dataset = Dataset.objects.get(name=name)
        except Dataset.DoesNotExist:
            raise DataError(f"Dataset '{name}' not found")
        
        # Drop from DuckDB
        safe_name = name.replace(' ', '_').lower()
        try:
            self.conn.execute(f"DROP TABLE IF EXISTS {safe_name}")
        except Exception:
            pass
        
        # Delete file
        if delete_files and os.path.exists(dataset.file_path):
            os.remove(dataset.file_path)
        
        dataset.delete()
        logger.info(f"Deleted dataset '{name}'")
