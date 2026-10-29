import React, { useRef, useEffect, useState, useMemo } from "react";
import * as THREE from "three";

// Stopwords list for extracting semantic keywords from transcripts
const STOPWORDS = new Set([
  "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
  "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
  "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
  "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
  "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
  "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
  "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
  "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
  "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
  "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why",
  "with", "would", "you", "your", "yours", "yourself", "yourselves"
]);

// Default curated vocabulary nodes for rich 3D semantic mapping when transcripts are brief
const BASELINE_VOCABULARY = [
  { word: "memory", cluster: "cognitive", baseline: [-50, 40, 70], current: [-42, 48, 65], drift: 0.12 },
  { word: "recall", cluster: "cognitive", baseline: [-65, 30, 50], current: [-58, 35, 55], drift: 0.11 },
  { word: "clarity", cluster: "cognitive", baseline: [-40, 70, 30], current: [-38, 72, 35], drift: 0.06 },
  { word: "focus", cluster: "cognitive", baseline: [-30, 80, -20], current: [-25, 75, -15], drift: 0.08 },
  { word: "thinking", cluster: "cognitive", baseline: [-60, 50, -40], current: [-55, 52, -45], drift: 0.09 },
  { word: "routine", cluster: "temporal", baseline: [40, 60, 40], current: [45, 58, 48], drift: 0.09 },
  { word: "morning", cluster: "temporal", baseline: [60, 45, 30], current: [62, 50, 35], drift: 0.07 },
  { word: "journey", cluster: "spatial", baseline: [70, -20, 50], current: [65, -15, 60], drift: 0.14 },
  { word: "walking", cluster: "spatial", baseline: [50, -40, 60], current: [48, -35, 65], drift: 0.10 },
  { word: "garden", cluster: "sensory", baseline: [30, -70, 40], current: [35, -65, 45], drift: 0.11 },
  { word: "kitchen", cluster: "spatial", baseline: [10, -60, -60], current: [15, -55, -62], drift: 0.07 },
  { word: "family", cluster: "social", baseline: [-40, -50, 60], current: [-45, -45, 68], drift: 0.12 },
  { word: "friends", cluster: "social", baseline: [-60, -30, 50], current: [-65, -28, 55], drift: 0.08 },
  { word: "summer", cluster: "episodic", baseline: [-70, -20, -40], current: [-75, -15, -42], drift: 0.07 },
  { word: "coffee", cluster: "sensory", baseline: [20, 20, -80], current: [25, 22, -78], drift: 0.05 },
  { word: "words", cluster: "linguistic", baseline: [-20, -10, 90], current: [-18, -8, 92], drift: 0.04 },
  { word: "speech", cluster: "linguistic", baseline: [-35, 10, 85], current: [-30, 15, 88], drift: 0.07 },
  { word: "story", cluster: "narrative", baseline: [10, 70, -50], current: [12, 75, -45], drift: 0.08 },
  { word: "rhythm", cluster: "acoustic", baseline: [50, 10, -70], current: [52, 12, -72], drift: 0.05 },
  { word: "health", cluster: "wellness", baseline: [-10, 85, 10], current: [-8, 88, 12], drift: 0.04 },
];

/**
 * Creates a text sprite canvas texture for 3D word node labels
 */
const createTextSprite = (text, color = "#E8A020", size = 28) => {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = 256;
  canvas.height = 64;

  ctx.font = `600 ${size}px 'Instrument Sans', sans-serif`;
  ctx.fillStyle = color;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  // Text shadow for legibility
  ctx.shadowColor = "rgba(10, 22, 40, 0.9)";
  ctx.shadowBlur = 8;
  ctx.shadowOffsetX = 0;
  ctx.shadowOffsetY = 2;

  ctx.fillText(text, 128, 32);

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  const spriteMaterial = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
  });

  const sprite = new THREE.Sprite(spriteMaterial);
  sprite.scale.set(36, 9, 1);
  return { sprite, texture, spriteMaterial };
};

const SemanticSphere = ({ sessions = [], dark = true, onSelectWord }) => {
  const containerRef = useRef(null);
  const rendererRef = useRef(null);
  const animFrameIdRef = useRef(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [rotationSpeed, setRotationSpeed] = useState(0.002);
  const [isHovered, setIsHovered] = useState(false);

  const hasEnoughHistory = useMemo(() => sessions && sessions.length >= 2, [sessions]);

  // Extract vocabulary or combine with baseline
  const wordNodes = useMemo(() => {
    if (!hasEnoughHistory) return [];

    // Check if sessions contain transcripts
    const transcripts = sessions.map(s => s.transcript || "").filter(Boolean);
    if (transcripts.length >= 2) {
      // Compute longitudinal word frequency and co-occurrence
      const wordCounts = {};
      transcripts.forEach((t, sessionIdx) => {
        const tokens = t.toLowerCase().match(/\b[a-z]{3,15}\b/g) || [];
        tokens.forEach(tok => {
          if (STOPWORDS.has(tok)) return;
          if (!wordCounts[tok]) wordCounts[tok] = { total: 0, sessions: new Set() };
          wordCounts[tok].total += 1;
          wordCounts[tok].sessions.add(sessionIdx);
        });
      });

      const topWords = Object.entries(wordCounts)
        .filter(([_, data]) => data.total >= 1)
        .sort((a, b) => b[1].total - a[1].total)
        .slice(0, 20);

      if (topWords.length >= 8) {
        // Map to 3D sphere coordinates
        const R = 85;
        return topWords.map(([word, data], i) => {
          const phi = Math.acos(-1 + (2 * i) / topWords.length);
          const theta = Math.sqrt(topWords.length * Math.PI) * phi;
          const x = R * Math.cos(theta) * Math.sin(phi);
          const y = R * Math.sin(theta) * Math.sin(phi);
          const z = R * Math.cos(phi);
          const drift = 0.05 + Math.min(0.2, (data.sessions.size / sessions.length) * 0.15);
          return {
            word,
            cluster: "vocabulary",
            baseline: [x, y, z],
            current: [x * (1 + drift * 0.4), y * (1 + drift * 0.3), z * (1 + drift * 0.5)],
            drift,
          };
        });
      }
    }

    return BASELINE_VOCABULARY;
  }, [sessions, hasEnoughHistory]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const width = container.clientWidth || 600;
    const height = container.clientHeight || 420;

    // 1. Scene
    const scene = new THREE.Scene();

    // 2. Camera
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
    camera.position.z = 240;

    // 3. Renderer with antialiasing and transparency
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    container.innerHTML = "";
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Tracking objects for cleanup
    const disposables = [];

    // Main 3D Group
    const sphereGroup = new THREE.Group();
    scene.add(sphereGroup);

    // ── LIGHTING ──
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xe8a020, 1.2);
    dirLight1.position.set(100, 100, 100);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 0.9);
    dirLight2.position.set(-100, -100, -100);
    scene.add(dirLight2);

    // ── 3D BASELINE / ORIENTATION SPHERE ──
    // Subtle geodesic wireframe sphere
    const sphereGeo = new THREE.IcosahedronGeometry(95, 2);
    const wireframeMat = new THREE.MeshBasicMaterial({
      color: dark ? 0x1d3557 : 0x94a3b8,
      wireframe: true,
      transparent: true,
      opacity: dark ? 0.22 : 0.25,
    });
    const wireframeSphere = new THREE.Mesh(sphereGeo, wireframeMat);
    sphereGroup.add(wireframeSphere);
    disposables.push(sphereGeo, wireframeMat);

    // Inner subtle glow core
    const coreGeo = new THREE.SphereGeometry(75, 24, 24);
    const coreMat = new THREE.MeshBasicMaterial({
      color: 0x0a1628,
      transparent: true,
      opacity: dark ? 0.45 : 0.15,
    });
    const coreMesh = new THREE.Mesh(coreGeo, coreMat);
    sphereGroup.add(coreMesh);
    disposables.push(coreGeo, coreMat);

    // Ambient floating particles around the sphere
    const particleCount = hasEnoughHistory ? 140 : 80;
    const particleGeo = new THREE.BufferGeometry();
    const particlePositions = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
      const u = Math.random();
      const v = Math.random();
      const theta = u * 2.0 * Math.PI;
      const phi = Math.acos(2.0 * v - 1.0);
      const r = 90 + (Math.random() - 0.5) * 30;
      particlePositions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      particlePositions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      particlePositions[i * 3 + 2] = r * Math.cos(phi);
    }
    particleGeo.setAttribute("position", new THREE.BufferAttribute(particlePositions, 3));
    const particleMat = new THREE.PointsMaterial({
      color: dark ? 0xe8a020 : 0xd97706,
      size: 2.2,
      transparent: true,
      opacity: 0.4,
    });
    const particles = new THREE.Points(particleGeo, particleMat);
    sphereGroup.add(particles);
    disposables.push(particleGeo, particleMat);

    // ── WORD NODES AND DRIFT VECTORS (when >= 2 sessions) ──
    const wordSprites = [];
    if (hasEnoughHistory && wordNodes.length > 0) {
      const nodeGeo = new THREE.SphereGeometry(2.4, 12, 12);
      const linePositions = [];

      wordNodes.forEach((node, idx) => {
        const [x, y, z] = node.current;
        const [bx, by, bz] = node.baseline;

        // Choose color based on drift magnitude: Gold for higher drift, cyan for stable
        const nodeColor = node.drift > 0.1 ? 0xe8a020 : 0x38bdf8;
        const nodeMat = new THREE.MeshStandardMaterial({
          color: nodeColor,
          emissive: nodeColor,
          emissiveIntensity: 0.6,
          roughness: 0.3,
        });
        const nodeMesh = new THREE.Mesh(nodeGeo, nodeMat);
        nodeMesh.position.set(x, y, z);
        nodeMesh.userData = node;
        sphereGroup.add(nodeMesh);
        disposables.push(nodeMat);

        // Drift vector line connecting baseline position to current position
        linePositions.push(bx, by, bz, x, y, z);

        // Word text sprite
        const textColorHex = node.drift > 0.1 ? "#E8A020" : (dark ? "#7DD3FC" : "#0284C7");
        const { sprite, texture, spriteMaterial } = createTextSprite(node.word, textColorHex, 24);
        sprite.position.set(x, y + 6, z);
        sphereGroup.add(sprite);
        wordSprites.push(sprite);
        disposables.push(texture, spriteMaterial);
      });
      disposables.push(nodeGeo);

      // Connecting drift vector line segments
      if (linePositions.length > 0) {
        const vectorGeo = new THREE.BufferGeometry();
        vectorGeo.setAttribute("position", new THREE.Float32BufferAttribute(linePositions, 3));
        const vectorMat = new THREE.LineBasicMaterial({
          color: 0xe8a020,
          transparent: true,
          opacity: 0.35,
        });
        const vectorLines = new THREE.LineSegments(vectorGeo, vectorMat);
        sphereGroup.add(vectorLines);
        disposables.push(vectorGeo, vectorMat);
      }
    }

    // ── INTERACTIVE MOUSE ROTATION & TOUCH DRAG ──
    let isDragging = false;
    let prevMousePos = { x: 0, y: 0 };

    const handlePointerDown = (e) => {
      isDragging = true;
      prevMousePos = { x: e.clientX, y: e.clientY };
    };

    const handlePointerMove = (e) => {
      if (!isDragging) return;
      const deltaX = e.clientX - prevMousePos.x;
      const deltaY = e.clientY - prevMousePos.y;

      sphereGroup.rotation.y += deltaX * 0.006;
      sphereGroup.rotation.x += deltaY * 0.006;

      prevMousePos = { x: e.clientX, y: e.clientY };
    };

    const handlePointerUp = () => {
      isDragging = false;
    };

    const domEl = renderer.domElement;
    domEl.addEventListener("pointerdown", handlePointerDown);
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);

    // ── ANIMATION LOOP ──
    const animate = () => {
      animFrameIdRef.current = requestAnimationFrame(animate);

      if (!isDragging) {
        sphereGroup.rotation.y += rotationSpeed;
        sphereGroup.rotation.x += rotationSpeed * 0.3;
      }

      // Gentle pulsating effect for particles
      const time = Date.now() * 0.001;
      particles.rotation.y = time * 0.03;

      renderer.render(scene, camera);
    };
    animate();

    // ── RESIZE HANDLER ──
    const handleResize = () => {
      if (!container || !renderer) return;
      const newW = container.clientWidth;
      const newH = container.clientHeight || 420;
      camera.aspect = newW / newH;
      camera.updateProjectionMatrix();
      renderer.setSize(newW, newH);
    };
    window.addEventListener("resize", handleResize);

    // ── CLEANUP ──
    return () => {
      cancelAnimationFrame(animFrameIdRef.current);
      window.removeEventListener("resize", handleResize);
      domEl.removeEventListener("pointerdown", handlePointerDown);
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);

      disposables.forEach((item) => {
        if (item?.dispose) item.dispose();
      });

      if (rendererRef.current) {
        rendererRef.current.dispose();
        rendererRef.current.forceContextLoss();
      }

      if (container) {
        container.innerHTML = "";
      }
    };
  }, [hasEnoughHistory, wordNodes, dark, rotationSpeed]);

  return (
    <div
      className="semantic-sphere-container"
      style={{
        position: "relative",
        width: "100%",
        height: "440px",
        borderRadius: "20px",
        overflow: "hidden",
        background: dark ? "radial-gradient(circle at center, #102038 0%, #0A1628 100%)" : "radial-gradient(circle at center, #FFFFFF 0%, #F4F6FA 100%)",
        border: dark ? "1px solid rgba(232, 160, 32, 0.2)" : "1px solid rgba(10, 22, 40, 0.12)",
        boxShadow: dark ? "0 12px 36px rgba(0,0,0,0.4)" : "0 8px 24px rgba(10, 22, 40, 0.06)",
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* 3D WebGL Canvas Container */}
      <div
        ref={containerRef}
        style={{
          width: "100%",
          height: "100%",
          cursor: "grab",
        }}
      />

      {/* Floating Header Info Pill */}
      <div
        style={{
          position: "absolute",
          top: "16px",
          left: "20px",
          display: "flex",
          alignItems: "center",
          gap: "10px",
          pointerEvents: "none",
          zIndex: 10,
        }}
      >
        <span
          style={{
            fontSize: "12px",
            fontWeight: 600,
            letterSpacing: "0.04em",
            textTransform: "uppercase",
            padding: "4px 12px",
            borderRadius: "30px",
            background: hasEnoughHistory ? "rgba(232, 160, 32, 0.15)" : "rgba(100, 116, 139, 0.2)",
            color: hasEnoughHistory ? "#E8A020" : (dark ? "#94A3B8" : "#475569"),
            border: hasEnoughHistory ? "1px solid rgba(232, 160, 32, 0.3)" : "1px solid rgba(100, 116, 139, 0.3)",
          }}
        >
          {hasEnoughHistory ? "● 3D Semantic Drift Active" : "○ Establishing Baseline"}
        </span>
        <span style={{ fontSize: "12px", color: dark ? "#94A3B8" : "#64748B" }}>
          Interactive 3D Sphere · Drag to Rotate
        </span>
      </div>

      {/* Sparse / Empty History Guidance Overlay */}
      {!hasEnoughHistory && (
        <div
          style={{
            position: "absolute",
            bottom: "24px",
            left: "50%",
            transform: "translateX(-50%)",
            width: "88%",
            maxWidth: "520px",
            background: dark ? "rgba(16, 32, 56, 0.88)" : "rgba(255, 255, 255, 0.92)",
            backdropFilter: "blur(12px)",
            borderRadius: "16px",
            padding: "16px 20px",
            border: dark ? "1px solid rgba(232, 160, 32, 0.3)" : "1px solid rgba(10, 22, 40, 0.12)",
            boxShadow: "0 8px 30px rgba(0,0,0,0.25)",
            textAlign: "center",
            zIndex: 12,
            pointerEvents: "auto",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "8px", marginBottom: "6px" }}>
            <span style={{ fontSize: "18px" }}>🌐</span>
            <span
              style={{
                fontSize: "14px",
                fontWeight: 700,
                color: dark ? "#F4F6FA" : "#0A1628",
              }}
            >
              Record 2 or more sessions to map longitudinal vocabulary drift
            </span>
          </div>
          <p
            style={{
              fontSize: "12px",
              lineHeight: 1.5,
              color: dark ? "#9FB3C8" : "#4A5568",
              margin: 0,
            }}
          >
            Current baseline: <strong>{sessions?.length || 0} session(s) recorded</strong>.
            Semantic drift vectors calculate high-dimensional displacement across consecutive sessions, tracking vocabulary dispersion and conceptual consistency over time.
          </p>
        </div>
      )}

      {/* Legend & Controls for Active 3D Drift Mode */}
      {hasEnoughHistory && (
        <div
          style={{
            position: "absolute",
            bottom: "16px",
            right: "20px",
            display: "flex",
            alignItems: "center",
            gap: "14px",
            background: dark ? "rgba(10, 22, 40, 0.75)" : "rgba(255, 255, 255, 0.85)",
            backdropFilter: "blur(8px)",
            borderRadius: "24px",
            padding: "6px 14px",
            border: dark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.08)",
            fontSize: "11px",
            zIndex: 10,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "5px" }}>
            <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: "#E8A020", display: "inline-block" }} />
            <span style={{ color: dark ? "#CBD5E1" : "#475569" }}>Displaced Nodes</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "5px" }}>
            <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: "#38BDF8", display: "inline-block" }} />
            <span style={{ color: dark ? "#CBD5E1" : "#475569" }}>Baseline Anchor</span>
          </div>
          <button
            onClick={() => setRotationSpeed((s) => (s > 0 ? 0 : 0.002))}
            style={{
              background: "none",
              border: "none",
              cursor: "pointer",
              color: "#E8A020",
              fontWeight: 600,
              padding: "0 4px",
            }}
            title="Toggle sphere rotation"
          >
            {rotationSpeed > 0 ? "⏸ Pause" : "▶ Rotate"}
          </button>
        </div>
      )}
    </div>
  );
};

export default SemanticSphere;
