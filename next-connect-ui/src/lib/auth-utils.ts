import { getServerSession } from "next-auth/next"
import { authOptions } from "@/lib/auth"
import { redirect } from "next/navigation"

// 서버 컴포넌트에서 인증 상태 확인 및 세션 가져오기
export async function getAuthSession() {
  return await getServerSession(authOptions)
}

// 인증이 필요한 서버 컴포넌트에서 사용
export async function requireAuth() {
  const session = await getAuthSession()

  if (!session) {
    redirect("/signin")
  }

  return session
}

// 비인증 상태가 필요한 서버 컴포넌트에서 사용 (로그인 페이지 등)
export async function requireGuest() {
  const session = await getAuthSession()

  if (session) {
    redirect("/")
  }
}

