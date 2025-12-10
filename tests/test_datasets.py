"""
Tests for the datasets app.
"""
import pytest
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import shutil
from io import BytesIO

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mlops_project'))

import django
django.setup()

from django.core.files.uploadedfile import SimpleUploadedFile
from datasets.models import Dataset
from datasets.services import DataLoader


@pytest.fixture
def temp_media_dir():
    """Create a temporary media directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_csv_file():
    """Create a sample CSV file for testing."""
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': [10.0, 20.0, 30.0, 40.0, 50.0],
        'col3': ['a', 'b', 'c', 'd', 'e'],
    })
    
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    
    return SimpleUploadedFile('test.csv', buffer.getvalue(), content_type='text/csv')


@pytest.fixture
def sample_parquet_file():
    """Create a sample Parquet file for testing."""
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    
    return SimpleUploadedFile('test.parquet', buffer.getvalue())


@pytest.mark.django_db
class TestDataLoader:
    """Tests for DataLoader service."""
    
    def test_upload_csv(self, sample_csv_file, temp_media_dir):
        """Test uploading a CSV file."""
        from django.conf import settings
        settings.MEDIA_ROOT = temp_media_dir
        
        loader = DataLoader(datasets_dir=os.path.join(temp_media_dir, 'datasets'))
        dataset = loader.upload(sample_csv_file, 'test_csv_dataset')
        
        assert dataset.name == 'test_csv_dataset'
        assert dataset.file_type == 'csv'
        assert dataset.row_count == 5
        assert len(dataset.column_names) == 3
        assert 'col1' in dataset.column_names
        
        # Cleanup
        dataset.delete()
    
    def test_upload_parquet(self, sample_parquet_file, temp_media_dir):
        """Test uploading a Parquet file."""
        from django.conf import settings
        settings.MEDIA_ROOT = temp_media_dir
        
        loader = DataLoader(datasets_dir=os.path.join(temp_media_dir, 'datasets'))
        dataset = loader.upload(sample_parquet_file, 'test_parquet_dataset', file_type='parquet')
        
        assert dataset.name == 'test_parquet_dataset'
        assert dataset.file_type == 'parquet'
        assert dataset.row_count == 5
        
        # Cleanup
        dataset.delete()
    
    def test_load_dataset(self, sample_csv_file, temp_media_dir):
        """Test loading a dataset."""
        from django.conf import settings
        settings.MEDIA_ROOT = temp_media_dir
        
        loader = DataLoader(datasets_dir=os.path.join(temp_media_dir, 'datasets'))
        loader.upload(sample_csv_file, 'load_test')
        
        # Load all data
        df = loader.load('load_test')
        assert len(df) == 5
        
        # Load with limit
        df_limited = loader.load('load_test', limit=2)
        assert len(df_limited) == 2
        
        # Load specific columns
        df_cols = loader.load('load_test', columns=['col1'])
        assert list(df_cols.columns) == ['col1']
        
        # Cleanup
        Dataset.objects.filter(name='load_test').delete()
    
    def test_get_preview(self, sample_csv_file, temp_media_dir):
        """Test getting dataset preview."""
        from django.conf import settings
        settings.MEDIA_ROOT = temp_media_dir
        
        loader = DataLoader(datasets_dir=os.path.join(temp_media_dir, 'datasets'))
        loader.upload(sample_csv_file, 'preview_test')
        
        preview = loader.get_preview('preview_test', max_rows=3)
        assert len(preview) == 3
        
        # Cleanup
        Dataset.objects.filter(name='preview_test').delete()
    
    def test_statistics_computed(self, sample_csv_file, temp_media_dir):
        """Test that statistics are computed on upload."""
        from django.conf import settings
        settings.MEDIA_ROOT = temp_media_dir
        
        loader = DataLoader(datasets_dir=os.path.join(temp_media_dir, 'datasets'))
        dataset = loader.upload(sample_csv_file, 'stats_test')
        
        assert dataset.statistics is not None
        assert 'col1' in dataset.statistics
        assert 'col2' in dataset.statistics
        
        # Numeric columns should have mean, std, etc.
        assert 'mean' in dataset.statistics['col1']
        assert 'mean' in dataset.statistics['col2']
        
        # Cleanup
        dataset.delete()
    
    def test_delete_dataset(self, sample_csv_file, temp_media_dir):
        """Test deleting a dataset."""
        from django.conf import settings
        settings.MEDIA_ROOT = temp_media_dir
        
        loader = DataLoader(datasets_dir=os.path.join(temp_media_dir, 'datasets'))
        dataset = loader.upload(sample_csv_file, 'delete_test')
        file_path = dataset.file_path
        
        # Verify file exists
        assert os.path.exists(file_path)
        
        # Delete
        loader.delete('delete_test')
        
        # Verify dataset record is gone
        assert not Dataset.objects.filter(name='delete_test').exists()
        
        # Verify file is deleted
        assert not os.path.exists(file_path)
    
    def test_sql_query(self, sample_csv_file, temp_media_dir):
        """Test SQL querying with DuckDB."""
        from django.conf import settings
        settings.MEDIA_ROOT = temp_media_dir
        
        loader = DataLoader(datasets_dir=os.path.join(temp_media_dir, 'datasets'))
        loader.upload(sample_csv_file, 'query_test')
        
        # Query
        result = loader.query("SELECT * FROM query_test WHERE col1 > 2")
        assert len(result) == 3  # col1 values 3, 4, 5
        
        # Cleanup
        Dataset.objects.filter(name='query_test').delete()
