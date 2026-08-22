import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { toast } from "sonner";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Rocket, PlusCircle, Loader2, Trash2, AlertTriangle } from "lucide-react";
import {
  useBaasProjects,
  useCreateBaasProject,
  useDeleteBaasProject,
} from "@/hooks/use-baas";
import { CopyableSecret } from "@/components/copyable-secret";
import { GradientIcon } from "@/components/gradient-icon";
import { EmptyState } from "@/components/empty-state";
import { requireAuth } from "@/lib/require-auth";
import { ApiError } from "@/lib/api-service";
import type { BaasProject, BaasProjectCreateResult } from "@/lib/api-types";

export const Route = createFileRoute("/baas/")({
  beforeLoad: requireAuth,
  component: BaasPage,
});

function CreateProjectDialog({ onCreated }: { onCreated: () => void }) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [result, setResult] = useState<BaasProjectCreateResult | null>(null);

  const createMutation = useCreateBaasProject();

  const resetAndClose = () => {
    setName("");
    setResult(null);
    setOpen(false);
  };

  const handleCreate = async () => {
    if (!name.trim()) {
      toast.error("Give your project a name.");
      return;
    }
    try {
      const res = await createMutation.mutateAsync(name.trim());
      setResult(res);
      onCreated();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to create project");
    }
  };

  return (
    <Dialog open={open} onOpenChange={(v) => (v ? setOpen(true) : result ? undefined : resetAndClose())}>
      <DialogTrigger asChild>
        <Button>
          <PlusCircle className="h-4 w-4 mr-1.5" /> New Backend Project
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>{result ? "Project created" : "New backend project"}</DialogTitle>
        </DialogHeader>

        {!result ? (
          <div className="space-y-4">
            <div className="space-y-1.5">
              <Label>Project name</Label>
              <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="My Ecommerce App" />
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-start gap-2 rounded-md border border-warning/30 bg-warning/10 px-3 py-2 text-sm text-warning">
              <AlertTriangle className="h-4 w-4 flex-shrink-0 mt-0.5" />
              <span>
                The secret key is shown only this once and cannot be retrieved again. Copy both keys now and
                store them somewhere safe.
              </span>
            </div>
            <CopyableSecret label="Public key (safe for client-side code)" value={result.public_key} />
            <CopyableSecret label="Secret key (server-side only — never expose this)" value={result.secret_key} />
          </div>
        )}

        <DialogFooter>
          {!result ? (
            <Button onClick={handleCreate} disabled={createMutation.isPending}>
              {createMutation.isPending && <Loader2 className="h-4 w-4 animate-spin mr-1.5" />}
              Create project
            </Button>
          ) : (
            <Button onClick={resetAndClose}>I've saved it — done</Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function ProjectCard({ project, onDeleted }: { project: BaasProject; onDeleted: () => void }) {
  const deleteMutation = useDeleteBaasProject();

  const handleDelete = async () => {
    try {
      await deleteMutation.mutateAsync(project.id);
      toast.success("Project and all its data deleted");
      onDeleted();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to delete project");
    }
  };

  return (
    <Card variant="glass" className="card-interactive">
      <CardContent className="p-4 space-y-3">
        <div className="flex items-start gap-3">
          <GradientIcon icon={Rocket} />
          <div className="min-w-0 flex-1">
            <p className="font-medium truncate">{project.name}</p>
            <p className="text-xs text-muted-foreground truncate">
              Created {new Date(project.created_at).toLocaleDateString()}
            </p>
          </div>
        </div>
        <div className="flex gap-2">
          <Button asChild size="sm" variant="outline" className="flex-1">
            <Link to="/baas/$projectId" params={{ projectId: project.id }}>
              Open
            </Link>
          </Button>
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button size="sm" variant="outline" className="text-destructive hover:text-destructive">
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Delete "{project.name}"?</AlertDialogTitle>
                <AlertDialogDescription>
                  This permanently deletes the project, all its API keys, table schemas, records, and
                  end-user accounts. This cannot be undone.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction onClick={handleDelete} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
                  Delete
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      </CardContent>
    </Card>
  );
}

function BaasPage() {
  const { data, isLoading, isError, refetch } = useBaasProjects();

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Backend Projects</h1>
          <p className="text-sm text-muted-foreground">
            Create a project to get a generic data API, keys, and end-user auth for your own app.
          </p>
        </div>
        <CreateProjectDialog onCreated={() => refetch()} />
      </div>

      {isLoading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-32 w-full" />
          ))}
        </div>
      ) : isError ? (
        <Card>
          <CardContent className="p-6 text-sm text-destructive">Couldn't load your projects.</CardContent>
        </Card>
      ) : !data || data.length === 0 ? (
        <Card>
          <CardContent className="p-10">
            <EmptyState
              icon={Rocket}
              title="No backend projects yet"
              description="Create a backend project to get a project ID, public/secret keys, and a generic data API."
            />
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {data.map((p) => (
            <ProjectCard key={p.id} project={p} onDeleted={() => refetch()} />
          ))}
        </div>
      )}
    </div>
  );
}
