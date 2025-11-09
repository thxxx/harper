// src/app/api/hello/route.ts
import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  return NextResponse.json({ message: "GET 요청 성공!" });
}

export async function POST(req: NextRequest) {
  const body = await req.json();
  return NextResponse.json({ message: "POST 요청 성공!", data: body });
}
