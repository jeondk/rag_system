"use client";

import Link from "next/link"
import { useRouter } from "next/navigation"
import { useForm } from "react-hook-form"
import { zodResolver } from "@hookform/resolvers/zod"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { useAuth } from "@/hooks/use-auth"
import { signupSchema, SignupFormValues } from "@/lib/schemas";
import { useState } from "react";
import { useTranslation } from "@/hooks/use-translation";

export default function SignUp() {
  const router = useRouter();
  const { register: registerUser, registerLoading } = useAuth();
  const [emailVerificationRequired, setEmailVerificationRequired] = useState(false);
  const { t } = useTranslation();
  
  // 1. useForm 설정
  const {
    register,
    handleSubmit,
    setError,
    formState: { errors, isSubmitting },
  } = useForm<SignupFormValues>({
    resolver: zodResolver(signupSchema),
    defaultValues: {
      email: "",
      password: "",
      confirmPassword: "",
    },
  });

  // 3. 폼 제출 로직 구현
  const onSubmit = async (data: SignupFormValues) => {
    try {
      // 4. useAuth의 register 함수 사용
      const result = await registerUser({
        email: data.email,
        password: data.password,
      });

      // 5. 결과 처리
      if (result.success) {
        // 이메일 확인이 필요한 경우
        if (result.message.includes('이메일') && result.message.includes('인증')) {
          setEmailVerificationRequired(true);
          toast.success(t('auth.signUpSuccess'));
        } else {
          // 일반적인 회원가입 성공
          toast.success(result.message);
          setTimeout(() => {
            router.push("/");
          }, 1000);
        }
      } else {
        // 서버 에러 처리
        setError("root", {
          type: "manual",
          message: result.message,
        });
      }
    } catch (error) {
      console.error("회원가입 오류:", error);
      setError("root", {
        type: "manual",
        message: "회원가입 처리 중 오류가 발생했습니다.",
      });
    }
  };

  // 이메일 확인 안내 화면
  if (emailVerificationRequired) {
    return (
      <div className="flex min-h-svh w-full items-center justify-center p-6 md:p-10 bg-background dark:bg-background">
        <Card className="w-full max-w-md">
          <CardHeader className="space-y-1 pb-2">
            <h1 className="text-2xl font-bold tracking-tight dark:text-gray-100">{t('auth.emailVerification.title')}</h1>
            <p className="text-sm text-muted-foreground dark:text-gray-300">{t('auth.emailVerification.subtitle')}</p>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg space-y-2">
              <p className="text-sm text-blue-900 dark:text-blue-200">
                <strong>{t('auth.emailVerification.message')}</strong>
              </p>
              <p className="text-sm text-blue-700 dark:text-blue-300">
                {t('auth.emailVerification.description')}
              </p>
            </div>
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground dark:text-gray-300 text-center">
                {t('auth.emailVerification.emailNotReceived')}
              </p>
            </div>
            <Button 
              className="w-full bg-black dark:bg-white dark:text-black text-white hover:bg-black/90 dark:hover:bg-white/90"
              onClick={() => router.push("/signin")}
            >
              {t('auth.emailVerification.goToLogin')}
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="flex min-h-svh w-full items-center justify-center p-6 md:p-10 bg-background dark:bg-background">
      <Card className="w-full max-w-md">
        <CardHeader className="space-y-1 pb-2">
          <h1 className="text-2xl font-bold tracking-tight dark:text-gray-100">{t('auth.signUp')}</h1>
          <p className="text-sm text-muted-foreground dark:text-gray-300">{t('auth.signUpDescription')}</p>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* 2. 폼 필드와 register 함수 연결 */}
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
            <div className="space-y-2">
              <label htmlFor="email" className="text-sm font-medium dark:text-gray-200">
                {t('auth.email')}
              </label>
              <Input 
                id="email" 
                type="email" 
                placeholder="user@example.com" 
                {...register("email")} 
              />
              {errors.email && (
                <p className="text-sm text-red-500 dark:text-red-400">{errors.email.message}</p>
              )}
            </div>
            <div className="space-y-2">
              <label htmlFor="password" className="text-sm font-medium dark:text-gray-200">
                {t('auth.password')}
              </label>
              <Input 
                id="password" 
                type="password" 
                {...register("password")} 
              />
              {errors.password && (
                <p className="text-sm text-red-500 dark:text-red-400">{errors.password.message}</p>
              )}
            </div>
            <div className="space-y-2">
              <label htmlFor="confirmPassword" className="text-sm font-medium dark:text-gray-200">
                {t('auth.confirmPassword')}
              </label>
              <Input 
                id="confirmPassword" 
                type="password" 
                {...register("confirmPassword")} 
              />
              {errors.confirmPassword && (
                <p className="text-sm text-red-500 dark:text-red-400">{errors.confirmPassword.message}</p>
              )}
            </div>
            
            {/* 루트 에러 메시지 표시 */}
            {errors.root && (
              <p className="text-sm text-red-500 dark:text-red-400 text-center">{errors.root.message}</p>
            )}
            
            <Button 
              type="submit" 
              className="w-full bg-black dark:bg-white dark:text-black text-white hover:bg-black/90 dark:hover:bg-white/90"
              disabled={isSubmitting || registerLoading}
            >
              {(isSubmitting || registerLoading) ? t('auth.signUpProcessing') : t('auth.signUpButton')}
            </Button>
          </form>
          <div className="text-center text-sm dark:text-gray-300">
            {t('auth.alreadyHaveAccount')}{" "}
            <Link href="/signin" className="font-medium underline dark:text-blue-400 hover:dark:text-blue-300">
              {t('auth.signIn')}
            </Link>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
