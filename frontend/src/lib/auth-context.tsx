import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { api } from "./api-service";
import {
  clearToken,
  getStoredUser,
  getToken,
  setStoredUser,
  setToken,
  type StoredUser,
} from "./auth-store";

type AuthContextValue = {
  user: StoredUser | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string, rememberMe?: boolean) => Promise<void>;
  signup: (name: string, email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<StoredUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Validate any stored token against the server on first load. A cached
  // user is shown optimistically so the UI doesn't flash empty while that
  // request is in flight.
  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      const token = getToken();
      if (!token) {
        setIsLoading(false);
        return;
      }

      const cached = getStoredUser();
      if (cached) setUser(cached);

      try {
        const me = await api.me();
        if (!cancelled) {
          setUser(me);
          setStoredUser(me);
        }
      } catch {
        if (!cancelled) {
          clearToken();
          setUser(null);
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }

    bootstrap();
    return () => {
      cancelled = true;
    };
  }, []);

  const login = async (email: string, password: string, rememberMe = false) => {
    const { user: loggedInUser, token } = await api.login({
      email,
      password,
      remember_me: rememberMe,
    });
    setToken(token);
    setStoredUser(loggedInUser);
    setUser(loggedInUser);
  };

  const signup = async (name: string, email: string, password: string) => {
    const { user: newUser, token } = await api.signup({ name, email, password });
    setToken(token);
    setStoredUser(newUser);
    setUser(newUser);
  };

  const logout = async () => {
    try {
      await api.logout();
    } catch {
      // Stateless JWT — the token is discarded client-side regardless of
      // whether this network call succeeds.
    }
    clearToken();
    setUser(null);
  };

  return (
    <AuthContext.Provider
      value={{ user, isAuthenticated: !!user, isLoading, login, signup, logout }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
