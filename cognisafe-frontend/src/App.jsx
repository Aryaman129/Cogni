import { useEffect } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import ProtectedRoute from "./components/common/ProtectedRoute";
import Auth      from "./pages/Auth";
import Dashboard from "./pages/Dashboard";
import Session   from "./pages/Session";
import Brain     from "./pages/Brain";
import ARReport  from "./pages/ARReport";

const App = () => {

  // Wake up HF Space as soon as app loads
  useEffect(() => {
    fetch("https://alamfarzann-cognisafe-ml.hf.space/health")
      .catch(() => {}); // silently ignore errors
  }, []);

  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/"     element={<Navigate to="/auth" replace />} />
          <Route path="/auth" element={<Auth />} />

          <Route path="/dashboard" element={
            <ProtectedRoute><Dashboard /></ProtectedRoute>
          } />
          <Route path="/session" element={
            <ProtectedRoute><Session /></ProtectedRoute>
          } />
          <Route path="/brain" element={
            <ProtectedRoute><Brain /></ProtectedRoute>
          } />
          <Route path="/ar-report" element={
            <ProtectedRoute><ARReport /></ProtectedRoute>
          } />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
};

export default App;