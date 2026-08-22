import { useNavigate } from "@tanstack/react-router";
import { ChevronsUpDown, LogOut } from "lucide-react";
import { toast } from "sonner";
import { SidebarMenuButton } from "@/components/ui/sidebar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useAuth } from "@/lib/auth-context";
import { cn } from "@/lib/utils";

function initials(name: string) {
  return name
    .split(" ")
    .filter(Boolean)
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase())
    .join("");
}

export function UserMenu({ variant }: { variant: "sidebar" | "topbar" }) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = async () => {
    await logout();
    toast.success("Logged out");
    navigate({ to: "/login" });
  };

  if (!user) return null;

  const avatar = (
    <div className="flex h-7 w-7 items-center justify-center rounded-full bg-[image:var(--gradient-brand)] text-xs font-semibold text-white flex-shrink-0">
      {initials(user.name) || "?"}
    </div>
  );

  const trigger =
    variant === "sidebar" ? (
      <SidebarMenuButton className="data-[state=open]:bg-accent">
        {avatar}
        <div className="flex flex-col min-w-0 text-left group-data-[collapsible=icon]:hidden">
          <span className="truncate text-sm font-medium">{user.name}</span>
          <span className="truncate text-xs text-muted-foreground">{user.email}</span>
        </div>
        <ChevronsUpDown className="ml-auto h-4 w-4 text-muted-foreground group-data-[collapsible=icon]:hidden" />
      </SidebarMenuButton>
    ) : (
      <button
        type="button"
        className={cn(
          "flex h-8 w-8 items-center justify-center rounded-full transition-opacity hover:opacity-80",
        )}
        aria-label="Account menu"
      >
        {avatar}
      </button>
    );

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>{trigger}</DropdownMenuTrigger>
      <DropdownMenuContent side={variant === "sidebar" ? "top" : "bottom"} align="end" className="w-56">
        <DropdownMenuLabel className="font-normal">
          <div className="flex flex-col space-y-1">
            <p className="text-sm font-medium leading-none">{user.name}</p>
            <p className="text-xs leading-none text-muted-foreground">{user.email}</p>
          </div>
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={handleLogout}>
          <LogOut className="mr-2 h-4 w-4" />
          Log out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
