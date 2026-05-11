import { useState, useEffect, useRef, useCallback } from "react";

// ── Shared data-fetching hook ──
export function useAPI(fetchFn, transformFn, deps = [], enabled = true) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(enabled);
  const [error, setError] = useState(null);
  const [refetchKey, setRefetchKey] = useState(0);
  const fetchedRef = useRef(false);

  const refetch = useCallback(() => {
    fetchedRef.current = false;
    setError(null);
    setRefetchKey(k => k + 1);
  }, []);

  useEffect(() => {
    if (!enabled || fetchedRef.current) return;
    fetchedRef.current = true;
    let cancelled = false;
    setLoading(true);
    fetchFn()
      .then(raw => { if (!cancelled) setData(transformFn ? transformFn(raw) : raw); })
      .catch(err => { if (!cancelled) setError(err.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, refetchKey, ...deps]);

  return { data, loading, error, refetch };
}
