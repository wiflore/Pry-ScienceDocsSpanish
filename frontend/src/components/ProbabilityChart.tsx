import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { T1_META } from "../lib/labels";
import type { EtiquetaT1 } from "../types/api";

interface Props {
  probabilidades: Record<string, number>;
  type: "t1" | "t2";
}

const T2_COLORS: Record<string, string> = {
  contribucion: "#EC4899",
  no_contribucion: "#9CA3AF",
};

export default function ProbabilityChart({ probabilidades, type }: Props) {
  const data = Object.entries(probabilidades).map(([code, p]) => ({
    code,
    label: type === "t1" ? code : code === "contribucion" ? "CONTR" : "NO",
    prob: +(p * 100).toFixed(2),
  }));

  const colorOf = (code: string) =>
    type === "t1"
      ? T1_META[code as EtiquetaT1]?.hex ?? "#9CA3AF"
      : T2_COLORS[code] ?? "#9CA3AF";

  return (
    <ResponsiveContainer width="100%" height={type === "t1" ? 180 : 100}>
      <BarChart data={data} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
        <XAxis
          dataKey="label"
          tick={{ fontSize: 11 }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          unit="%"
          tick={{ fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          width={40}
        />
        <Tooltip
          formatter={(value) => [`${value}%`, "Probabilidad"]}
          contentStyle={{
            background: "white",
            border: "1px solid #e5e7eb",
            borderRadius: 6,
            fontSize: 12,
          }}
        />
        <Bar dataKey="prob" radius={[4, 4, 0, 0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={colorOf(d.code)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
