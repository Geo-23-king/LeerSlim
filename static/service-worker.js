const CACHE_NAME = "leerslim-v1";
const ASSETS = [
  "/",
  "/static/style.css",
  "/static/manifest.json",
  "/static/leerslim-logo1.png"
];

self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(ASSETS))
  );
});

self.addEventListener("fetch", event => {
  event.respondWith(
    caches.match(event.request).then(response => {
      return response || fetch(event.request);
    })
  );
});