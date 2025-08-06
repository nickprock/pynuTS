import pytest
import numpy as np
from pynuTS.clustering import ScalableDTWKMeans, ClusteringMetrics


def test_clustering_basic_fit_predict():
    """
    Test 1: Test base di fit e predict
    
    Verifica che il clustering funzioni con dati semplici:
    - Crea dati con pattern chiari
    - Fa il fit del modello
    - Verifica che predict funzioni
    """
    # Crea dati semplici con 2 pattern chiari
    data = []
    
    # Pattern 1: serie crescenti
    for i in range(10):
        series = np.array([1, 2, 3, 4, 5]) + np.random.normal(0, 0.1, 5)
        data.append(series)
    
    # Pattern 2: serie decrescenti  
    for i in range(10):
        series = np.array([5, 4, 3, 2, 1]) + np.random.normal(0, 0.1, 5)
        data.append(series)
    
    # Crea il clusterer
    clusterer = ScalableDTWKMeans(n_clusters=2, random_state=42)
    
    # Test fit
    clusterer.fit(data)
    
    # Verifiche base
    assert clusterer.labels_ is not None
    assert len(clusterer.labels_) == 20  # 20 serie totali
    assert len(set(clusterer.labels_)) <= 2  # massimo 2 cluster
    assert clusterer.cluster_centers_ is not None
    assert len(clusterer.cluster_centers_) == 2  # 2 centroidi
    
    # Test predict
    test_data = [np.array([1, 2, 3, 4, 5]), np.array([5, 4, 3, 2, 1])]
    predictions = clusterer.predict(test_data)
    
    assert len(predictions) == 2
    assert all(pred in [0, 1] for pred in predictions)
    
    print("✓ Test 1 passed: Basic fit and predict work correctly")


def test_clustering_metrics():
    """
    Test 2: Test delle metriche di clustering
    
    Verifica che le metriche vengano calcolate correttamente:
    - Crea e fa il fit di un modello
    - Ottiene le metriche
    - Verifica che abbiano valori sensati
    """
    # Dati semplici
    np.random.seed(42)
    data = [np.random.randn(10) for _ in range(15)]
    
    # Fit del modello
    clusterer = ScalableDTWKMeans(n_clusters=3, max_iter=10, random_state=42)
    clusterer.fit(data)
    
    # Test metriche
    metrics = clusterer.get_clustering_metrics()
    
    # Verifiche
    assert isinstance(metrics, ClusteringMetrics)
    assert metrics.inertia > 0  # L'inerzia dovrebbe essere positiva
    assert metrics.computation_time >= 0  # Il tempo dovrebbe essere non-negativo
    assert metrics.convergence_iterations > 0  # Almeno 1 iterazione
    assert isinstance(metrics.inertia, float)
    
    # Verifica che l'inerzia sia ragionevole (non infinita)
    assert metrics.inertia < float('inf')
    
    print(f"✓ Test 2 passed: Metrics calculated correctly")
    print(f"  Inertia: {metrics.inertia:.2f}")
    print(f"  Iterations: {metrics.convergence_iterations}")
    print(f"  Time: {metrics.computation_time:.3f}s")


def test_clustering_different_data_types():
    """
    Test 3: Test con diversi tipi di dati
    
    Verifica che il clustering gestisca correttamente:
    - Array NumPy
    - Liste Python
    - Dati di lunghezze diverse
    """
    # Prepara dati di tipi diversi
    data = []
    
    # NumPy arrays
    data.append(np.array([1.0, 2.0, 3.0, 4.0]))
    data.append(np.array([4.0, 3.0, 2.0, 1.0]))
    
    # Liste Python
    data.append([2, 3, 4, 5])
    data.append([5, 4, 3, 2])
    
    # Lunghezze diverse (il clustering dovrebbe gestirle)
    data.append([1, 2, 3])  # Più corta
    data.append([1, 2, 3, 4, 5, 6])  # Più lunga
    
    # Test clustering
    clusterer = ScalableDTWKMeans(n_clusters=2, max_iter=5, random_state=42)
    
    try:
        clusterer.fit(data)
        labels = clusterer.labels_
        
        # Verifiche base
        assert labels is not None
        assert len(labels) == 6  # 6 serie totali
        assert all(label in [0, 1] for label in labels)  # Solo cluster 0 e 1
        
        # Test che i centroidi esistano
        assert clusterer.cluster_centers_ is not None
        assert len(clusterer.cluster_centers_) == 2
        
        # Test predict con dati nuovi
        new_data = [[1, 2, 3, 4], [4, 3, 2, 1]]
        predictions = clusterer.predict(new_data)
        assert len(predictions) == 2
        
        print("✓ Test 3 passed: Different data types handled correctly")
        print(f"  Processed {len(data)} series of varying types and lengths")
        print(f"  Final labels: {labels}")
        
    except Exception as e:
        pytest.fail(f"Test failed with error: {e}")


if __name__ == "__main__":
    """
    Esegue i 3 test semplici in sequenza
    """
    print("Esecuzione test semplici per clustering.py")
    print("=" * 50)
    
    try:
        # Test 1
        test_clustering_basic_fit_predict()
        
        # Test 2  
        test_clustering_metrics()
        
        # Test 3
        test_clustering_different_data_types()
        
        print("=" * 50)
        print("✅ Tutti i test sono passati con successo!")
        
    except Exception as e:
        print(f"❌ Test fallito: {e}")
        raise
    
    except AssertionError as e:
        print(f"❌ Asserzione fallita: {e}")
        raise