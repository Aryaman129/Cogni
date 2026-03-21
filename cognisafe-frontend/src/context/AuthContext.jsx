import { createContext, useContext, useState, useEffect } from "react";

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user,    setUser]    = useState(null);
  const [token,   setToken]   = useState(() => localStorage.getItem("cog_jwt"));
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // on app load, restore user from localStorage
    const savedUser = localStorage.getItem("cog_user");
    if (savedUser && token) {
      try { setUser(JSON.parse(savedUser)); }
      catch { logout(); }
    }
    setLoading(false);
  }, []);

  const login = (tokenData) => {
    // tokenData = { access_token, user_id, name, email }
    localStorage.setItem("cog_jwt",  tokenData.access_token);
    localStorage.setItem("cog_user", JSON.stringify({
      id:    tokenData.user_id,
      name:  tokenData.name,
      email: tokenData.email,
    }));
    setToken(tokenData.access_token);
    setUser({ id: tokenData.user_id, name: tokenData.name, email: tokenData.email });
  };

  const logout = () => {
    localStorage.removeItem("cog_jwt");
    localStorage.removeItem("cog_user");
    localStorage.removeItem("cog_last_session");
    localStorage.removeItem("cog_last_result_tier");
    setToken(null);
    setUser(null);
  };

  const isLoggedIn = !!token && !!user;

  return (
    <AuthContext.Provider value={{ user, token, login, logout, isLoggedIn, loading }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside AuthProvider");
  return ctx;
};

export default AuthContext;