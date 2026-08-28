import { useEffect, useRef, useState } from "react";

/**
 * Tracks whether an element has scrolled into the viewport, for
 * scroll-reveal animations. Reveals once and stays revealed — re-hiding
 * content every time it scrolls out feels distracting on a data-heavy app
 * rather than a marketing page.
 *
 * Respects prefers-reduced-motion by reporting "already in view"
 * immediately, skipping the observer entirely.
 */
export function useInView<T extends HTMLElement = HTMLDivElement>(
  options?: IntersectionObserverInit,
) {
  const ref = useRef<T>(null);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setInView(true);
      return;
    }

    // Content already on-screen at mount (e.g. above the fold on load)
    // reveals immediately rather than waiting on the observer's first
    // async callback — both because it reads better (no flash of
    // pointlessly-hidden content the user could already see) and because
    // it's a reliable fallback in any environment where
    // IntersectionObserver callbacks are delayed or suppressed (e.g. a
    // background/non-composited tab).
    const rect = el.getBoundingClientRect();
    if (rect.top < window.innerHeight && rect.bottom > 0) {
      setInView(true);
      return;
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setInView(true);
          observer.disconnect();
        }
      },
      { threshold: 0.15, rootMargin: "0px 0px -40px 0px", ...options },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return { ref, inView };
}
