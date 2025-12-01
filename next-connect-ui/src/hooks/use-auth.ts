import { useSession, signIn, signOut } from "next-auth/react";
import { useState, useEffect } from "react";
import { useTranslation } from "./use-translation";

// 회원가입 요청 타입 정의
interface RegisterParams {
  email: string;
  password?: string;
}

// 회원가입 응답 타입 정의
interface RegisterResponse {
  success: boolean;
  message: string;
}

// 로그인 요청 타입 정의
interface LoginParams {
  email?: string;
  password?: string;
}

// 로그인 응답 타입 정의
interface LoginResponse {
  success: boolean;
  message: string;
}

// 로그아웃 응답 타입 정의
interface LogoutResponse {
  success: boolean;
}

export const useAuth = () => {
  const { data: session, status, update } = useSession();
  const { t } = useTranslation();
  const isLoading = status === "loading";
  const isAuthenticated = status === "authenticated";
  const [registerLoading, setRegisterLoading] = useState(false);
  const [loginLoading, setLoginLoading] = useState(false);
  const [logoutLoading, setLogoutLoading] = useState(false);

  // Check for refresh token errors
  useEffect(() => {
    if (session?.error === "RefreshAccessTokenError") {
      // Token refresh failed, sign out the user
      signOut({ callbackUrl: '/signin' });
    }
  }, [session?.error]);

  // 회원가입 함수
  const register = async (params: RegisterParams): Promise<RegisterResponse> => {
    setRegisterLoading(true);
    try {
      // next-auth의 signIn 함수 호출
      const result = await signIn('credentials', {
        // credential일 경우 email, password 전달
        ...(params.email && params.password && {
          email: params.email,
          password: params.password,
          type: 'signup',
        }),
        // 리다이렉트 방지 (직접 처리하기 위함)
        redirect: false,
        // 로그인 성공 후 리다이렉트할 URL
        callbackUrl: '/',
      });

      // 로그인 결과 처리
      if (result?.error) {
        // 이메일 확인이 필요한 경우
        if (result.error === 'EMAIL_VERIFICATION_REQUIRED') {
          return {
            success: true,
            message: t('auth.emailVerification.message'),
          };
        }
        // 이미 가입된 사용자인 경우
        if (result.error === 'USER_ALREADY_EXISTS') {
          return {
            success: false,
            message: t('auth.signUpEmailExists'),
          };
        }
        
        return {
          success: false,
          message: t('auth.signUpError'),
        };
      }

      if (result?.url) {
        return {
          success: true,
          message: t('auth.signUpSuccess'),
        };
      }

      return {
        success: false,
        message: t('auth.unknownError'),
      };
    } finally {
      setRegisterLoading(false);
    }
  };

  // 로그인 함수
  const login = async (params: LoginParams): Promise<LoginResponse> => {
    setLoginLoading(true);
    try {
      // next-auth의 signIn 함수 호출
      const result = await signIn('credentials', {
        // credential일 경우 email, password 전달
        ...(params.email && params.password && {
          email: params.email,
          password: params.password,
        }),
        // 리다이렉트 방지 (직접 처리하기 위함)
        redirect: false,
        // 로그인 성공 후 리다이렉트할 URL
        callbackUrl: '/',
      });

      // 로그인 결과 처리
      if (result?.error) {
        return {
          success: false,
          message: result.error === 'CredentialsSignin' 
            ? t('auth.signInInvalidCredentials') 
            : t('auth.signInError'),
        };
      }

      if (result?.url) {
        return {
          success: true,
          message: t('auth.signInSuccess'),
        };
      }

      return {
        success: false,
        message: t('auth.unknownError'),
      };
    } catch (error) {
      console.error('Login error:', error);
      return {
        success: false,
        message: t('auth.signInError'),
      };
    } finally {
      setLoginLoading(false);
    }
  };
  
  // 로그아웃 함수
  const logout = async (): Promise<LogoutResponse> => {
    setLogoutLoading(true);
    try {
      // 1. 서버 세션 정리 (백엔드 API 호출)
      await fetch('/api/auth/logout', {
        method: 'POST',
      });

      // 2. NextAuth 세션 정리 (JWT 쿠키 삭제)
      await signOut({ 
        redirect: false,
        callbackUrl: '/'
      });
      
      return {
        success: true
      };
    } catch (error) {
      console.error('Logout error:', error);
      return {
        success: false
      };
    } finally {
      setLogoutLoading(false);
    }
  };

  return {
    session,
    user: session?.user,
    isLoading,
    isAuthenticated,
    signIn,
    signOut,
    update,
    register,
    registerLoading,
    login,
    loginLoading,
    logout,
    logoutLoading,
  };
}; 